"""Centralized constants definition"""
import numpy as np

# ========== Task Metadata ==========
# task_name -> prompt, robot_type mapping
TASK_METADATA = {
    "put_pen_into_pencil_case": {
        "prompt": "place the pen on the table into the pencil case",
        "robot_type": "aloha",
    },
    "move_objects_into_box": {
        "prompt": "place all the clutter on the desk into the white box",
        "robot_type": "franka",
    },
    "hang_toothbrush_cup": {
        "prompt": "hang the orange toothbrush cup on the cup holder",
        "robot_type": "ur5",
    },
    "place_shoes_on_rack": {
        "prompt": "place these shoes on the shoe rack",
        "robot_type": "arx5",
    },
    "clean_dining_table": {
        "prompt": "place all the trash into the green trash bin, and put the dishes into the transparent basket",
        "robot_type": "aloha",
    },
    "plug_in_network_cable": {
        "prompt": "Insert the RJ45 connector of the Ethernet cable into the host.",
        "robot_type": "aloha",
    },
    "pour_fries_into_plate": {
        "prompt": "open the box lid and pour the chips from the box onto the plate.",
        "robot_type": "aloha",
    },
    "scan_QR_code": {
        "prompt": "scan the QR code on the medicine box using the scanner",
        "robot_type": "aloha",
    },
    "stick_tape_to_box": {
        "prompt": "tear off a piece of clear tape and stick it onto the metal box",
        "robot_type": "aloha",
    },
    "press_three_buttons": {
        "prompt": "press the pink, blue, and green buttons in sequence",
        "robot_type": "franka",
    },
    "turn_on_faucet": {
        "prompt": "grasp the faucet switch and turn it on",
        "robot_type": "aloha",
    },
    "search_green_boxes": {
        "prompt": "search through the stack of blocks for the green blocks and place it into the yellow box",
        "robot_type": "arx5",
    },
    "stack_bowls": {
        "prompt": "stack the two smaller bowls on top of the largest bowl one by one.",
        "robot_type": "aloha",
    },
    "open_the_drawer": {
        "prompt": "open the drawer",
        "robot_type": "arx5",
    },
    "turn_on_light_switch": {
        "prompt": "turn on the light switch",
        "robot_type": "arx5",
    },
    "put_cup_on_coaster": {
        "prompt": "place the cup on the coaster",
        "robot_type": "arx5",
    },
    "wipe_the_table": {
        "prompt": "pull out a tissue, wipe the stains on the table clean, and then discard the tissue into the trash bin",
        "robot_type": "arx5",
    },
    "make_vegetarian_sandwich": {
        "prompt": "make a vegetable sandwich",
        "robot_type": "aloha",
    },
    "put_opener_in_drawer": {
        "prompt": "place the can opener into the right-hand drawer",
        "robot_type": "aloha",
    },
    "water_potted_plant": {
        "prompt": "water the potted plant using the kettle",
        "robot_type": "arx5",
    },
    "fold_dishcloth": {
        "prompt": "fold the dishcloth in half twice, then place it in the position slightly to the front and left",
        "robot_type": "arx5",
    },
    "sweep_the_rubbish": {
        "prompt": "sweep the trash into the dustpan using a broom",
        "robot_type": "aloha",
    },
    "arrange_paper_cups": {
        "prompt": "stack the four paper cups on top of the paper cup closest to the shelf one by one and place the stacked cups on the shelf",
        "robot_type": "arx5",
    },
    "arrange_flowers": {
        "prompt": "insert the three flowers on the table into the vase one by one",
        "robot_type": "arx5",
    },
    "arrange_fruits_in_basket": {
        "prompt": "Place the four fruits into the nearby basket one by one",
        "robot_type": "ur5",
    },
    "set_the_plates": {
        "prompt": "place the three plates onto the plate rack one by one",
        "robot_type": "ur5",
    },
    "shred_scrap_paper": {
        "prompt": "place the white paper on the shelf into the shredder for shredding",
        "robot_type": "ur5",
    },
    "sort_books": {
        "prompt": "place the three books on the shelf into the corresponding book sections of the black bookshelf",
        "robot_type": "ur5",
    },
    "sort_electronic_products": {
        "prompt": "classify the four electronic products on the table and place them into the corresponding transparent baskets",
        "robot_type": "arx5",
    },
    "stack_color_blocks": {
        "prompt": "stack the yellow block on top of the orange block",
        "robot_type": "ur5",
    },
}

# ========== Robot Configuration ==========
# robot_type -> image_type list
IMAGE_TYPE_MAP = {
    "arx5": ["high", "left_hand", "right_hand"],
    "aloha": ["high", "left_hand", "right_hand"],
    "ur5": ["right_hand", "left_hand"],
    "franka": ["high", "left_hand", "right_hand"],
}

# robot_type -> image_key mapping (for model input)
IMAGE_MAPPING = {
    "arx5": {"high": "image_1", "left_hand": "image_2", "right_hand": "image_0"},
    "aloha": {"high": "image_0", "left_hand": "image_1", "right_hand": "image_2"},
    "ur5": {"right_hand": "image_0", "left_hand": "image_1"},
    "franka": {"high": "image_1", "left_hand": "image_2", "right_hand": "image_0"},
}

# ========== Joint Limits ==========
ANGLE2RAD = np.pi / 180.0
ALOHA_JOINT_MIN = [
    -150 * ANGLE2RAD, 0 * ANGLE2RAD, -170 * ANGLE2RAD, -100 * ANGLE2RAD, -70 * ANGLE2RAD, -180 * ANGLE2RAD, 0,
    -150 * ANGLE2RAD, 0 * ANGLE2RAD, -170 * ANGLE2RAD, -99 * ANGLE2RAD, -70 * ANGLE2RAD, -180 * ANGLE2RAD, 0,
]
ALOHA_JOINT_MAX = [
    150 * ANGLE2RAD, 180 * ANGLE2RAD, 0 * ANGLE2RAD, 100 * ANGLE2RAD, 70 * ANGLE2RAD, 180 * ANGLE2RAD, 0.1,
    150 * ANGLE2RAD, 180 * ANGLE2RAD, 0 * ANGLE2RAD, 99 * ANGLE2RAD, 70 * ANGLE2RAD, 180 * ANGLE2RAD, 0.1,
]
