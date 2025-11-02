from os import stat
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


from enum import Enum, auto
from grid_universe.levels.factories import create_hazard
from grid_universe.components.properties.appearance import AppearanceName


class TestLevel[Enum]:
    Level_3 = 1
    Level_4 = auto()
    Level_5 = auto()
    Level_6_1 = auto()
    Level_6_2 = auto()
    Level_7 = auto()
    Level_8_1 = auto()
    Level_8_2 = auto()
    Level_8_3 = auto()
    Level_9 = auto()
    Level_9_1 = auto()
    Level_11_1 = auto()
    Level_11_2 = auto()
    Level_12 = auto()


def generate_state() -> State:
    if CURRENT_LEVEL == TestLevel.Level_3:
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
    elif CURRENT_LEVEL == TestLevel.Level_4:
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

    elif CURRENT_LEVEL == TestLevel.Level_5:
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
    elif CURRENT_LEVEL == TestLevel.Level_6_1:
        level = Level(
            width=8,
            height=1,
            move_fn=default_move_fn,
            objective_fn=exit_objective_fn,  # reach the exit to win
            seed=22,
        )

        for y in range(8):
            for x in range(1):
                level.add((y, x), create_floor(cost_amount=3))

        portal = create_portal()

        level.add((0, 0), create_agent())  # 5 health by default
        level.add(
            (2, 0), create_hazard(appearance=AppearanceName.SPIKE, damage=2)
        )  # the spike deals 2 damage
        level.add((4, 0), create_exit())

        state = to_state(level)
        return state

    elif CURRENT_LEVEL == TestLevel.Level_6_2:
        level = Level(
            width=11,
            height=9,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=exit_objective_fn,  # reach the exit to win
            seed=23,  # for reproducibility
        )

        # 2) Layout: floors, then place objects
        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor())

                if x == 4 and y not in (0, 4, 8):
                    level.add((x, y), create_wall())

        level.add((1, 4), create_agent(health=5))
        level.add((9, 4), create_exit())
        level.add(
            (4, 4), create_hazard(appearance=AppearanceName.LAVA, damage=2, lethal=True)
        )  # You will die on contact with lava

        # 3) Convert to runtime State (immutable)
        state = to_state(level)
        return state

    elif CURRENT_LEVEL == TestLevel.Level_7:
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
    elif CURRENT_LEVEL == TestLevel.Level_8_1:
        from grid_universe.levels.factories import create_box
        from grid_universe.components.properties import MovingAxis

        level = Level(
            3, 3, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=18
        )
        for y in range(3):
            for x in range(3):
                level.add((x, y), create_floor())

        for y, x in ((0, 0), (0, 2), (2, 0), (2, 2)):
            level.add((x, y), create_wall())

        level.add((1, 0), create_agent())
        level.add((1, 2), create_exit())

        box = create_box(
            pushable=False,
            moving_axis=MovingAxis.HORIZONTAL,
            moving_direction=1,
            moving_bounce=True,
            moving_speed=1,
        )

        level.add((1, 1), box)

        state = to_state(level)
        return state
    elif CURRENT_LEVEL == TestLevel.Level_8_2:
        from grid_universe.levels.factories import create_box
        from grid_universe.components.properties.moving import MovingAxis

        level = Level(
            8, 1, move_fn=default_move_fn, objective_fn=exit_objective_fn, seed=19
        )
        for y in range(8):
            for x in range(1):
                level.add((y, x), create_floor())

        portal = create_portal()

        level.add((0, 0), create_agent())
        level.add((1, 0), create_box())
        level.add((2, 0), portal)
        level.add((6, 0), create_portal(pair=portal))
        level.add((3, 0), create_exit())

        box = create_box(
            pushable=False,
            moving_axis=MovingAxis.HORIZONTAL,
            moving_direction=1,
            moving_bounce=True,
            moving_speed=1,
        )

        state = to_state(level)
        return state
    elif CURRENT_LEVEL == TestLevel.Level_8_3:
        from grid_universe.levels.factories import create_box
        from grid_universe.components.properties.moving import MovingAxis

        level = Level(
            width=11,
            height=9,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=exit_objective_fn,  # win when stand on exit
            seed=20,  # for reproducibility
        )

        # 2) Layout: floors, then place objects
        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor())

                if x == 5 and y != 4:
                    level.add((x, y), create_wall())

        level.add((1, 4), create_agent(health=5))
        level.add((9, 4), create_exit())
        level.add((4, 4), create_box())

        # 3) Convert to runtime State (immutable)
        state = to_state(level)
        return state
    elif CURRENT_LEVEL == TestLevel.Level_9:
        from grid_universe.components.properties import MovingAxis
        from grid_universe.levels.factories import create_monster

        level = Level(
            width=13,
            height=9,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=exit_objective_fn,  # reach the exit to win
            seed=24,  # for reproducibility
        )

        # 2) Layout: floors, then place objects
        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor())

                if x in (6, 7) and y not in (4, 5):
                    level.add((x, y), create_wall())

        level.add((2, 4), create_agent(health=5))
        level.add((11, 4), create_exit())

        level.add(
            (6, 4),
            create_monster(
                lethal=True, moving_axis=MovingAxis.VERTICAL, moving_direction=1
            ),
        )
        level.add(
            (7, 4),
            create_monster(
                lethal=True, moving_axis=MovingAxis.VERTICAL, moving_direction=1
            ),
        )

        # 3) Convert to runtime State (immutable)
        state = to_state(level)
        return state
    elif CURRENT_LEVEL == TestLevel.Level_9_1:
        from grid_universe.levels.factories import create_monster
        from grid_universe.components.properties.pathfinding import PathfindingType

        level = Level(
            width=7,
            height=5,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=exit_objective_fn,  # reach the exit to win
            seed=25,  # for reproducibility
        )

        # 2) Layout: floors, then place objects
        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor())

        agent = create_agent()

        level.add((1, 2), agent)
        level.add((5, 4), create_exit())

        level.add(
            (0, 1),
            create_monster(
                lethal=True, pathfind_target=agent, path_type=PathfindingType.PATH
            ),
        )

        # 3) Convert to runtime State (immutable)
        state = to_state(level)
        return state
    elif CURRENT_LEVEL == TestLevel.Level_11_1:
        from grid_universe.levels.factories import (
            create_phasing_effect,
            create_speed_effect,
            create_immunity_effect,
        )

        level = Level(
            width=13,
            height=9,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=exit_objective_fn,  # win when stand on exit
            seed=123,  # for reproducibility
        )

        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor())

                if x == 6:
                    if y == 4:
                        level.add((x, y), create_door(key_id="my_key"))
                    else:
                        level.add((x, y), create_wall())

        level.add((1, 4), create_agent())
        level.add((11, 4), create_exit())
        level.add(
            (2, 1), create_phasing_effect(time=5)
        )  # For 5 turns, can walk through walls

        state = to_state(level)
        return state
    elif CURRENT_LEVEL == TestLevel.Level_11_2:
        from grid_universe.levels.factories import create_immunity_effect

        level = Level(
            width=13,
            height=9,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=exit_objective_fn,  # win when stand on exit
            seed=123,  # for reproducibility
        )

        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor(cost_amount=3))

                if x == 6:
                    if y == 4:
                        level.add(
                            (x, y),
                            create_hazard(
                                appearance=AppearanceName.LAVA, damage=5, lethal=True
                            ),
                        )
                    else:
                        level.add((x, y), create_wall())

        level.add((1, 4), create_agent())
        level.add((11, 4), create_exit())
        level.add(
            (2, 1), create_immunity_effect(usage=1)
        )  # You arre immune to damage once.

        state = to_state(level)
        return state
    elif CURRENT_LEVEL == TestLevel.Level_12:
        from grid_universe.components.properties import MovingAxis
        from grid_universe.levels.factories import create_speed_effect, create_monster

        level = Level(
            width=13,
            height=9,
            move_fn=default_move_fn,  # choose movement semantics
            objective_fn=exit_objective_fn,  # win when stand on exit
            seed=123,  # for reproducibility
        )

        for y in range(level.height):
            for x in range(level.width):
                level.add((x, y), create_floor())

                if 6 <= x <= 8:
                    if y == 4:
                        level.add(
                            (x, y),
                            create_monster(
                                lethal=True,
                                moving_axis=MovingAxis.VERTICAL,
                                moving_direction=1,
                            ),
                        )
                    elif y == 5:
                        continue
                    else:
                        level.add((x, y), create_wall())

        level.add((1, 4), create_agent())
        level.add((11, 4), create_exit())
        level.add(
            (5, 5), create_speed_effect(multiplier=2, time=3)
        )  # For 3 turns, your speed is doubled

        state = to_state(level)
        return state

    raise RuntimeError("Cannot find level")


CURRENT_LEVEL = TestLevel.Level_12
MAX_NUMBER_OF_STEPS: int = 50


def test():
    state = generate_state()

    agent = Agent()
    current_state = state

    action_list: List[Action] = []

    for _ in range(0, MAX_NUMBER_OF_STEPS):
        new_action = agent.step(from_state(current_state))
        current_state = step(current_state, new_action)
        action_list.append(new_action)
        print(current_state.score)
        print(new_action, "HERE BRO")
        render_state(current_state)

        if current_state.win:
            print("YOU WON BRO")
            exit(0)

    print("YOU LOST BRO")


test()
