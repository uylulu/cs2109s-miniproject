import random
import time
from typing import List
from queue import PriorityQueue
from grid_universe.moves import default_move_fn
from grid_universe.state import State
from grid_universe.actions import Action
from grid_universe.levels.grid import EntitySpec, Level
from PIL import Image

from grid_universe.objectives import (
    exit_objective_fn,
    default_objective_fn,
)
# from ciphertext.ciphertext_decoder import CiphertextDecoder
# from image_classification.classification_lib.image_classification import ImageClassify
# from image_classification.classification_lib.direction_classfication import (
#     DirectionClassify,
# )


from grid_universe.step import EntityID, step
from grid_universe.levels.convert import to_state

from grid_universe.components.properties import Position
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

# Asset root for rendering. You can change this if you want to use custom game assets.
from grid_universe.components.properties.moving import MovingAxis
from grid_universe.components.properties.appearance import AppearanceName

# Rendering and display
from grid_universe.renderer.texture import TextureRenderer


# Default renderer used throughout the notebook unless overridden in a cell
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
        self.max_steps_left: int = 0

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
        return self.steps_left > self.max_steps_left / 2

    def astar(self, state: "State"):
        initial_node = Node(state, None, None, 0, self.objective)

        if initial_node in self.good_action and self.can_exploit():
            self.steps_left -= 1
            return self.good_action[initial_node]

        self.good_action: "dict[Node, Action]" = {}
        self.steps_left = 0
        self.max_steps_left = 0

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
                if node.state.win or len(end_nodes) >= 4:
                    break
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
        self.max_steps_left = curr_node.current_step
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

    def parse_image(self, state: "Observation") -> "Level":
        width = state["info"]["config"]["width"]
        height = state["info"]["config"]["height"]

        image = state["image"]
        image_height, image_width, channel_size = image.shape
        grid_box_height, grid_box_width = image_height // height, image_width // width

        agent_info = state["info"]["agent"]
        if state["info"]["config"]["objective_fn"] == "default_objective_fn":
            self.objective = "default"
        elif state["info"]["config"]["objective_fn"] == "exit_objective_fn":
            self.objective = "exit"
        else:
            self.ciphertext_decoder = CiphertextDecoder()
            self.objective = self.ciphertext_decoder.predict(state["info"]["message"])
        res_level = Level(
            ### CONFIG Info
            width=width,
            height=height,
            move_fn=default_move_fn,
            objective_fn=default_objective_fn
            if self.objective == "default"
            else exit_objective_fn,
            seed=state["info"]["config"]["seed"],
            turn_limit=state["info"]["config"]["turn_limit"],
        )
        portal_count = 0
        lst_portal: None | EntitySpec = None
        for i in range(height):
            for j in range(width):
                y0 = i * grid_box_height
                y1 = (i + 1) * grid_box_height
                x0 = j * grid_box_width
                x1 = (j + 1) * grid_box_width

                grid_box = image[y0:y1, x0:x1, :]
                image_box = Image.fromarray(grid_box)
                pred = self.image_model.predict(image_box)
                if pred == "human" or pred == "sleeping":
                    res_level.add((j, i), create_agent(agent_info["health"]["health"]))
                elif pred == "wall":
                    res_level.add((j, i), create_wall())
                elif pred == "floor":
                    res_level.add((j, i), create_floor(FLOOR_TILE_COST))
                elif pred == "exit":
                    res_level.add((j, i), create_exit())
                elif pred == "coin":
                    res_level.add((j, i), create_coin(COIN_REWARD))
                elif pred == "gem":
                    res_level.add((j, i), create_core(reward=0, required=True))
                elif pred == "lava":
                    res_level.add(
                        (j, i),
                        create_hazard(AppearanceName.LAVA, damage=HAZARDS_DAMMAGE),
                    )
                elif pred == "box":
                    res_level.add((j, i), create_box(pushable=True))
                elif pred == "metalbox":
                    direction = self.direction_model.predict(image_box)

                    if direction == "no_direction":
                        res_level.add((j, i), create_box(pushable=False))
                    elif direction == "right":
                        res_level.add(
                            (j, i),
                            create_box(
                                pushable=False,
                                moving_axis=MovingAxis.HORIZONTAL,
                                moving_direction=1,
                            ),
                        )
                    elif direction == "left":
                        res_level.add(
                            (j, i),
                            create_box(
                                pushable=False,
                                moving_axis=MovingAxis.HORIZONTAL,
                                moving_direction=-1,
                            ),
                        )
                    elif direction == "up":
                        res_level.add(
                            (j, i),
                            create_box(
                                pushable=False,
                                moving_axis=MovingAxis.VERTICAL,
                                moving_direction=-1,
                            ),
                        )
                    else:
                        res_level.add(
                            (j, i),
                            create_box(
                                pushable=False,
                                moving_axis=MovingAxis.VERTICAL,
                                moving_direction=1,
                            ),
                        )
                elif pred == "key":
                    res_level.add((j, i), create_key("default_id"))
                elif pred == "locked":
                    res_level.add((j, i), create_door("default_id"))
                elif pred == "spike":
                    res_level.add(
                        (j, i),
                        create_hazard(
                            appearance=AppearanceName.SPIKE, damage=HAZARDS_DAMMAGE
                        ),
                    )
                elif pred == "robot":
                    direction = self.direction_model.predict(image_box)

                    if direction == "no_direction":
                        res_level.add((j, i), create_monster(damage=ENEMIES_DAMAGE))
                    elif direction == "right":
                        res_level.add(
                            (j, i),
                            create_monster(
                                damage=ENEMIES_DAMAGE,
                                moving_axis=MovingAxis.HORIZONTAL,
                                moving_direction=1,
                            ),
                        )
                    elif direction == "left":
                        res_level.add(
                            (j, i),
                            create_monster(
                                damage=ENEMIES_DAMAGE,
                                moving_axis=MovingAxis.HORIZONTAL,
                                moving_direction=-1,
                            ),
                        )
                    elif direction == "up":
                        res_level.add(
                            (j, i),
                            create_monster(
                                damage=ENEMIES_DAMAGE,
                                moving_axis=MovingAxis.VERTICAL,
                                moving_direction=-1,
                            ),
                        )
                    else:
                        res_level.add(
                            (j, i),
                            create_monster(
                                damage=ENEMIES_DAMAGE,
                                moving_axis=MovingAxis.VERTICAL,
                                moving_direction=1,
                            ),
                        )

                elif pred == "portal":
                    if portal_count > 2:
                        res_level.add((j, i), create_floor(FLOOR_TILE_COST))
                    elif lst_portal is None:
                        lst_portal = create_portal()
                        portal_count += 1
                        res_level.add((j, i), lst_portal)
                    elif lst_portal is not None:
                        portal_count += 1
                        res_level.add((j, i), create_portal(pair=lst_portal))
                elif pred == "ghost":
                    res_level.add((j, i), create_phasing_effect(time=POWERUP_DURATION))
                elif pred == "shield":
                    res_level.add(
                        (j, i), create_immunity_effect(usage=POWERUP_DURATION)
                    )
                elif pred == "boots":
                    res_level.add(
                        (j, i), create_speed_effect(multiplier=2, time=POWERUP_DURATION)
                    )
                elif pred == "opened":
                    res_level.add((j, i), create_floor(FLOOR_TILE_COST))
                else:
                    res_level.add((j, i), create_floor(FLOOR_TILE_COST))
        return res_level

    def step(self, state: "Level | Observation") -> "Action":
        self.time_limit = 25
        self.step_limit = 20
        if not isinstance(state, Level):
            if self.current_state is not None:
                action = self.astar(self.current_state)
                self.current_state = step(self.current_state, action)

                return action

            self.direction_model = DirectionClassify()
            self.image_model = ImageClassify()

            current_level = self.parse_image(state)
            self.current_state = to_state(current_level)

            action = self.astar(self.current_state)
            self.current_state = step(self.current_state, action)
            return action
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
