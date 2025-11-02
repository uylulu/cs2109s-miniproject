import subprocess
import tempfile

# Asset root for rendering. You can change this if you want to use custom game assets.
ASSET_ROOT = "data/assets/"

# Unified imports for Grid Universe tutorial (run this cell first)
from re import L
from typing import List, Tuple

# Core API
from grid_universe.levels.grid import Level
from grid_universe.state import State
from grid_universe.levels.convert import to_state, from_state
from grid_universe.actions import Action
from grid_universe.step import step

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

# Movement and objectives
from grid_universe.moves import default_move_fn
from grid_universe.objectives import (
    exit_objective_fn,
    default_objective_fn,
    all_pushable_at_exit_objective_fn,
    all_unlocked_objective_fn,
)

# Components and enums
from grid_universe.components.properties import Moving
from grid_universe.components.properties.moving import MovingAxis
from grid_universe.components.properties.appearance import AppearanceName
from grid_universe.components.properties.pathfinding import PathfindingType

# Rendering and display
from grid_universe.renderer.texture import TextureRenderer
from IPython.display import display
from numpy import void

# Default renderer used throughout the notebook unless overridden in a cell
renderer = TextureRenderer(resolution=240, asset_root=ASSET_ROOT)
renderer_large = TextureRenderer(resolution=480, asset_root=ASSET_ROOT)
from agent import Agent
from grid_universe.levels.factories import create_core
from grid_universe.levels.grid import Level
from grid_universe.levels.factories import create_door, create_key
from grid_universe.levels.factories import create_portal


def render_state(state: State) -> None:
    image = renderer_large.render(state)
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        image.save(tmp.name)
        subprocess.run(["kitty", "+kitten", "icat", tmp.name])


CURRENT_LEVEL: int = 5


def generate_state() -> State:
    if CURRENT_LEVEL == 3:
        level = Level(
            width=9,
            height=7,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=default_objective_fn,  # win when collecting all cores and standing on exit
            seed=14,  # for reproducibility
        )

        # 2) Layout: floors, then place objects
        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor())

                if y == 0 or y == level.height - 1 or x == 0 or x == level.width - 1:
                    level.add((x, y), create_wall())

        level.add((1, 1), create_agent(health=5))
        level.add((7, 5), create_exit())

        level.add((4, 1), create_wall())
        level.add((4, 2), create_wall())
        level.add((4, 4), create_wall())
        level.add((4, 5), create_wall())

        level.add((3, 2), create_core())

        # 3) Convert to runtime State (immutable)
        state = to_state(level)
        return state
    elif CURRENT_LEVEL == 4:
        level = Level(
            width=11,
            height=9,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=default_objective_fn,  # win when collecting all cores and standing on exit
            seed=15,  # for reproducibility
        )

        # 2) Layout: floors, then place objects
        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor())

                if y == 0 or y == level.height - 1 or x == 0 or x == level.width - 1:
                    level.add((x, y), create_wall())

        level.add((1, 4), create_agent(health=5))
        level.add((9, 4), create_exit())

        level.add((5, 1), create_core())
        level.add((5, 7), create_core())

        for y in range(1, 8):
            for x in range(1, 10):
                if y == 4 or x == 5:
                    continue

                level.add((x, y), create_wall())

        # 3) Convert to runtime State (immutable)
        state = to_state(level)
        return state

    elif CURRENT_LEVEL == 5:
        level = Level(
            width=11,
            height=9,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=exit_objective_fn,  # win when stand on exit
            seed=16,  # for reproducibility
        )

        # 2) Layout: floors, then place objects
        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor())

                if x == 5 and y != 4:
                    level.add((x, y), create_wall())

        level.add((1, 4), create_agent(health=5))
        level.add((9, 4), create_exit())

        level.add((5, 4), create_door(key_id="my_key"))
        level.add((2, 3), create_key(key_id="my_key"))

        # 3) Convert to runtime State (immutable)
        state = to_state(level)
        return state
    elif CURRENT_LEVEL == 7:
        level = Level(
            width=11,
            height=9,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=exit_objective_fn,  # win when stand on exit
            seed=17,  # for reproducibility
        )

        # 2) Layout: floors, then place objects
        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor())

                if 3 <= x <= 7 and y == 3:
                    level.add((x, y), create_wall())

        level.add((1, 4), create_agent(health=5))
        level.add((9, 4), create_exit())

        portal = create_portal()
        level.add((2, 1), portal)
        level.add((10, 4), create_portal(pair=portal))

        # 3) Convert to runtime State (immutable)
        state = to_state(level)
        return state

    raise RuntimeError("Cannot find level")


MAX_NUMBER_OF_STEPS: int = 50


def test():
    state = generate_state()

    agent = Agent()
    current_state = state

    action_list: List[Action] = []

    for i in range(0, MAX_NUMBER_OF_STEPS):
        new_action = agent.step(from_state(current_state))
        current_state = step(current_state, new_action)
        action_list.append(new_action)

        print(new_action, "HERE BRO")
        render_state(current_state)

        if current_state.win:
            print("YOU WON BRO")
            exit(0)

    print("YOU LOST BRO")


test()
