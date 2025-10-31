# Asset root for rendering. You can change this if you want to use custom game assets.
ASSET_ROOT = "data/assets/"

# Unified imports for Grid Universe tutorial (run this cell first)
from typing import List, Tuple

# Core API
from grid_universe.levels.grid import Level
from grid_universe.state import State
from grid_universe.levels.convert import to_state, from_state
from grid_universe.actions import Action
from grid_universe.step import step

# Factories
from grid_universe.levels.factories import (
    create_floor, create_agent, create_box, create_coin, create_exit, create_wall,
    create_key, create_door, create_portal, create_core, create_hazard, create_monster,
    create_phasing_effect, create_speed_effect, create_immunity_effect,
)

# Movement and objectives
from grid_universe.moves import default_move_fn
from grid_universe.objectives import (
    exit_objective_fn, default_objective_fn, all_pushable_at_exit_objective_fn, all_unlocked_objective_fn,
)

# Components and enums
from grid_universe.components.properties import Moving
from grid_universe.components.properties.moving import MovingAxis
from grid_universe.components.properties.appearance import AppearanceName
from grid_universe.components.properties.pathfinding import PathfindingType

# Rendering and display
from grid_universe.renderer.texture import TextureRenderer
from IPython.display import display
from grid_universe.levels.factories import create_wall

# Default renderer used throughout the notebook unless overridden in a cell
renderer = TextureRenderer(resolution=240, asset_root=ASSET_ROOT)
renderer_large = TextureRenderer(resolution=480, asset_root=ASSET_ROOT)

from queue import PriorityQueue
import random, time

class Node:
    state: State 
    parent_node: "Node | None"
    parent_action: Action | None
    current_step: int
   
    def __init__(self, state: State, parent_node: "Node | None", parent_action: Action | None, current_step: int):
        self.current_step = current_step
        self.state = state
        self.parent_node = parent_node
        self.parent_action = parent_action
        
    def __lt__(self, other: "Node"):
        return self.heuristic() < other.heuristic()
        
    def __eq__(self, value: object) -> bool:
        if type(value) is not Node:
            return False
        
        return self.state == value.state  
    
    def manhatan_dis(self, pos_1: tuple[int, int], pos_2: tuple[int, int]):
        return abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])
    
    def heuristic(self):
        return self.manhatan_dis(self.get_agent_pos(), self.get_exit_position()) - self.state.score
    
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
    
    def step(self, action: Action) -> State:
        agent_id = next(iter(self.state.agent.keys()))
        return step(self.state, action, agent_id)
     
    def __hash__(self) -> int:
        return hash((self.state.position))

MOVE_ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.PICK_UP]

class Agent:
    def __init__(self):
        self.distance: dict[Node, int] = {}
        self.good_action: dict[Node, Action | None] = {}
        self.step_limit: int = 0
        
    def get_exit_position(self, state: State) -> tuple[int, int]:
        exit_id = next(iter(state.exit.keys()))
        position = state.position.get(exit_id)
        if position is not None:
            return (position.x, position.y)
        
        raise Exception("Cannot find exit position")
    
    def get_agent_position(self, state: State) -> tuple[int, int]:
        agent_id = next(iter(state.agent.keys()))
        position = state.position.get(agent_id)
        if position is not None:
            return (position.x, position.y)
        
        raise Exception("Cannot find exit position")
        
    def is_end_node(self, node: Node) -> bool:
        return node.get_agent_pos() == node.get_exit_position() or node.current_step >= self.step_limit
   
    def astar(self, state: State):
        pq: PriorityQueue[Node] = PriorityQueue()
        vis: dict[Node, bool] = {}
        
        initial_node = Node(state, None, None, 0)
        pq.put(initial_node)
        
        end_nodes: List[Node] = []
        while not pq.empty():
            node = pq.get()
            
            if node in vis:
                continue
            
            vis[node] = True
            if self.is_end_node(node):
                end_nodes.append(node)
                continue
            
            for action in MOVE_ACTIONS:
                new_state = node.step(action)
                new_node = Node(new_state, node, action, node.current_step + 1)

                pq.put(new_node)
                
        return self.find_base_action(end_nodes)
                
    def find_base_action(self, end_nodes: List[Node]) -> Action:
        if len(end_nodes) == 0:
            random.seed(time.time_ns())
            return random.choice(MOVE_ACTIONS)
    
        best = min(end_nodes)
        lst_action: None | Action = None
        
        curr_node = best
        while curr_node.parent_action is not None and curr_node.parent_node is not None:
            lst_action = curr_node.parent_action
            curr_node = curr_node.parent_node
            
        if lst_action is None:
            random.seed(time.time_ns())
            return random.choice(MOVE_ACTIONS)
        
        return lst_action               
    
    def step(self, state: Level) -> Action:
        self.step_limit = 5
        
        current_state = to_state(state)
        return self.astar(current_state)

def test():
    from grid_universe.levels.factories import create_coin

    level = Level(
        width=9,
        height=7,
        move_fn=default_move_fn,           # choose movement semantics
        objective_fn=exit_objective_fn,    # win when stand on exit
        seed=13,                          # for reproducibility
    )

    # 2) Layout: floors, then place objects
    for y in range(level.height):
        for x in range(level.width):
            level.add((x, y), create_floor())

            if y == 0 or y == level.height - 1 or x == 0 or x == level.width - 1:
                level.add((x, y), create_wall())

    level.add((1, 1), create_agent(health=5))
    level.add((7, 5), create_exit())

    level.add((2, 2), create_wall())
    level.add((3, 2), create_wall())
    level.add((4, 2), create_wall())
    level.add((5, 2), create_wall())
    level.add((6, 2), create_wall())

    level.add((2, 4), create_wall())
    level.add((3, 4), create_wall())
    level.add((5, 4), create_wall())
    level.add((6, 4), create_wall())

    level.add((2, 1), create_coin(reward=10))
    level.add((4, 1), create_coin(reward=10))
    level.add((6, 1), create_coin(reward=10))

    # 3) Convert to runtime State (immutable)
    state = to_state(level)
    renderer_large.render(state)
    
    action_sequence: List[Action] = [] # Fill in the sequence of actions here!

    display(renderer_large.render(state))
    agent = Agent()
    current_state = state
    while not current_state.win:
        new_action = agent.step(from_state(current_state))
        current_state = step(current_state, new_action, next(iter(current_state.agent.keys())))
        action_sequence.append(new_action)   

    agent_id = next(iter(state.agent.keys()))

    for a in action_sequence:
        state = step(state, a, agent_id)

    display(renderer_large.render(state))

    if (not state.win):
        print("The agent has not reached the exit!")
    elif (state.score < 18):
        print("The agent does not reach the exit with minimal cost")
    else:
        print("Congratulations! The agent reaches the exit with the minimum cost.")


test()
