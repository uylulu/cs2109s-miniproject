# Unified imports for Grid Universe tutorial (run this cell first)
import random
import time
from typing import List
from queue import PriorityQueue
from grid_universe.moves import default_move_fn
from grid_universe.state import State
from grid_universe.actions import Action
from grid_universe.levels.grid import Level

from grid_universe.objectives import (
    exit_objective_fn,
    default_objective_fn,
)
from ciphertext.ciphertext_decoder import CiphertextDecoder
from image_classification.classification_lib.image_classification import ImageClassify


from grid_universe.step import EntityID, step
from grid_universe.levels.convert import to_state, from_state

from grid_universe.components.properties import Position, inventory
from grid_universe.levels.convert import _entity_object_from_state
from grid_universe.utils.ecs import entities_with_components_at

# Factories
from grid_universe.levels.factories import (
    create_floor,
    create_agent,
    create_box,
    create_coin,
    create_exit,
    create_wall,
    create_key,
    create_door,
    create_portal,
    create_core,
    create_hazard,
    create_monster,
    create_phasing_effect,
    create_speed_effect,
    create_immunity_effect,
)

# Core API
from grid_universe.gym_env import Observation

import torch
from torch import Tensor, nn

CIPHER_TEXT_MODEL_PATH = "ciphertext-model"


class Node:
    state: "State"
    parent_node: "Node | None"
    parent_action: "Action | None"
    current_step: int
    objective: str

    def __init__(
        self,
        state: "State",
        parent_node: "Node | None",
        parent_action: "Action | None",
        current_step: int,
        objective: str,
    ):
        self.current_step = current_step
        self.state = state
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.objective = objective

    def __lt__(self, other: "Node"):
        return self.f() < other.f()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Node):
            return False

        return self.__hash__() == value.__hash__()

    def manhatan_dis(self, pos_1: tuple[int, int], pos_2: tuple[int, int]):
        return abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])

    def get_agent_id(self) -> EntityID:
        return next(iter(self.state.agent.keys()))

    def check_locks(self) -> bool:
        agent_id = self.get_agent_id()
        pos = self.state.position.get(agent_id)
        if pos is not None:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                adjacent = Position(pos.x + dx, pos.y + dy)
                locked_items = entities_with_components_at(
                    self.state, adjacent, self.state.locked
                )
                if len(locked_items) > 0:
                    return True

        return False

    def check_inventory(self) -> bool:
        agent_id = self.get_agent_id()
        inventory = self.state.inventory.get(agent_id)

        if inventory is not None and len(inventory.item_ids) > 0:
            return True

        return False

    def get_cores(self) -> "List[tuple[int, int]]":
        collectibles = self.state.collectible
        cores: "List[tuple[int, int]]" = []

        for key, _ in collectibles.items():
            entity_spec = _entity_object_from_state(self.state, key)
            if entity_spec.required is not None:
                position = self.state.position.get(key)
                if position is None:
                    raise RuntimeError("Cannot find position of core")

                cores.append((position.x, position.y))
        return cores

    def f(self):
        return 3 * self.heuristic() - self.state.score

    def heuristic(self):
        cores = self.get_cores()
        if self.state.lose:
            return 1000000000000000000

        if self.objective == "default":
            if len(cores) == 0:
                return self.manhatan_dis(self.get_agent_pos(), self.get_exit_position())
            else:
                min_dis = 100000000000000
                agent_pos = self.get_agent_pos()
                for x, y in cores:
                    min_dis = min(
                        min_dis, abs(x - agent_pos[0]) + abs(y - agent_pos[1])
                    )

                return min_dis + len(cores) * 100000
        else:
            return self.manhatan_dis(self.get_agent_pos(), self.get_exit_position())

    def get_agent_pos(self) -> tuple[int, int]:
        position = self.state.position.get(next(iter(self.state.agent.keys())))

        if position is None:
            raise RuntimeError("Cannot find agent's position")

        return (position.x, position.y)

    def get_exit_position(self) -> tuple[int, int]:
        position = self.state.position.get(next(iter(self.state.exit.keys())))

        if position is None:
            raise RuntimeError("Cannot find agent's position")

        return (position.x, position.y)

    def step(self, action: "Action") -> "State":
        agent_id = next(iter(self.state.agent.keys()))
        return step(self.state, action, agent_id)

    def __hash__(self) -> int:
        agent_id = next(iter(self.state.agent.keys()))
        return hash(
            (
                self.state.position,
                self.state.inventory.get(agent_id),
                frozenset(self.state.locked.items()),  # adds door/lock awareness
                frozenset(
                    self.state.collectible.items()
                ),  # adds collectibles awareness
            )
        )


MOVE_ACTIONS = [
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    Action.PICK_UP,
    Action.USE_KEY,
    Action.WAIT,
]
POWERUP_DURATION: int = 5
HAZARDS_DAMMAGE: int = 2
ENEMIES_DAMAGE: int = 1
FLOOR_TILE_COST: int = 3
COIN_REWARD: int = 5
COIRE_REWARD: int = 0


class Agent:
    def __init__(self):
        self.distance: "dict[Node, int]" = {}
        self.good_action: "dict[Node, Action]" = {}
        self.step_limit: int = 0
        self.current_state: "State | None" = None
        self.time_limit: float = 0
        self.base_time: float = time.time()
        self.objective: str = "default"
        self.steps_left: int = (
            0  # records the number of steps left that we have explored
        )

    def get_exit_position(self, state: "State") -> tuple[int, int]:
        exit_id = next(iter(state.exit.keys()))
        position = state.position.get(exit_id)
        if position is not None:
            return (position.x, position.y)

        raise Exception("Cannot find exit position")

    def get_agent_position(self, state: "State") -> tuple[int, int]:
        agent_id = next(iter(state.agent.keys()))
        position = state.position.get(agent_id)
        if position is not None:
            return (position.x, position.y)

        raise Exception("Cannot find exit position")

    def is_end_node(self, node: "Node") -> bool:
        return (
            node.get_agent_pos() == node.get_exit_position()
            or node.current_step >= self.step_limit
        )

    # Stochastic function to decide whether to explore or exploit
    def can_exploit(self) -> bool:
        # time_left = self.get_time_left()
        return self.steps_left >= self.step_limit / 2

    def astar(self, state: "State"):
        initial_node = Node(state, None, None, 0, self.objective)

        if initial_node in self.good_action and self.can_exploit():
            self.steps_left -= 1
            return self.good_action[initial_node]

        self.good_action: "dict[Node, Action]" = {}
        self.steps_left = 0

        pq: PriorityQueue[Node] = PriorityQueue()
        pq.put(initial_node)

        cnt: int = 0

        vis: dict[Node, bool] = {}
        end_nodes: "List[Node]" = []
        while not pq.empty():
            node = pq.get()
            cnt += 1
            if node in vis:
                continue
            agent_pos_x, agent_pos_y = node.get_agent_pos()

            vis[node] = True
            if self.is_end_node(node):
                end_nodes.append(node)
                continue

            collectible_ids = entities_with_components_at(
                node.state, Position(agent_pos_x, agent_pos_y), node.state.collectible
            )

            for action in MOVE_ACTIONS:
                if (
                    (action == Action.PICK_UP and len(collectible_ids) > 0)
                    or (
                        action == Action.USE_KEY
                        and node.check_inventory()
                        and node.check_locks()
                    )
                    or (action not in (Action.PICK_UP, Action.USE_KEY))
                ):
                    new_state = node.step(action)
                    new_node = Node(
                        new_state, node, action, node.current_step + 1, self.objective
                    )
                    if new_node not in vis:
                        pq.put(new_node)
        self.steps_left = self.step_limit
        return self.find_base_action(end_nodes)

    def find_base_action(self, end_nodes: "List[Node]") -> "Action":
        if len(end_nodes) == 0:
            random.seed(time.time_ns())
            return random.choice(MOVE_ACTIONS)

        best = min(end_nodes)
        lst_action: "Action | None" = None

        curr_node = best
        while curr_node.parent_action is not None and curr_node.parent_node is not None:
            lst_action = curr_node.parent_action
            self.good_action[curr_node.parent_node] = lst_action
            curr_node = curr_node.parent_node

        if lst_action is None:
            random.seed(time.time_ns())
            return random.choice(MOVE_ACTIONS)

        return lst_action

    def get_time_left(self) -> float:
        return self.time_limit - (time.time() - self.base_time)

    def get_object_from_pred(self, pred: str, x: int, y: int):
        if pred == "boots":
            return

    def parse_image(self, state: "Observation") -> "Level":
        width = state["info"]["config"]["width"]
        height = state["info"]["config"]["height"]

        image = state["image"]
        image_height, image_width = image.shape
        grid_box_height, grid_box_width = image_height / height, image_width / width

        if state["info"]["config"]["objective_fn"] == "default":
            self.objective = "default"
        elif state["info"]["config"]["objective_fn"] == "exit":
            self.objective = "exit"
        else:
            self.ciphertext_decoder = CiphertextDecoder()
            self.objective = self.ciphertext_decoder.predict(state["info"]["message"])

        res_level = Level(
            width=width,
            height=height,
            move_fn=default_move_fn,
            objective_fn=default_objective_fn
            if self.objective == "default"
            else exit_objective_fn,
        )

        for i in range(0, height):
            for j in range(0, width):
                grid_box = image[i : i + grid_box_height, j : j + grid_box_width, :3]
                x = torch.Tensor(
                    grid_box,
                )
                pred = self.image_model.predict(x)

                # if

    def step(self, state: "Level | Observation") -> "Action":
        if not isinstance(state, Level):
            self.image_model = ImageClassify()
            return Action.DOWN
        elif isinstance(state, Level):
            self.time_limit = 25
            self.step_limit = 20

            action: Action

            if self.current_state is None:
                current_state = to_state(state)

                if current_state.objective_fn.__name__ == "default_objective_fn":
                    self.objective = "default"
                elif current_state.objective_fn.__name__ == "exit_objective_fn":
                    self.objective = "exit"
                elif current_state.message is not None:
                    self.ciphertext_decoder = CiphertextDecoder()
                    self.objective = self.ciphertext_decoder.predict(
                        current_state.message
                    )
                else:
                    self.objective = "default"

                action = self.astar(current_state)
                self.current_state = step(current_state, action)
            else:
                action = self.astar(self.current_state)
                self.current_state = step(self.current_state, action)

            return action
