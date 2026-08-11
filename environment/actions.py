from enum import Enum
from typing import Dict, Any

class ActionType(Enum):
    TERMINATE = 0
    REDUCE_CONV1 = 1
    REDUCE_CONV2 = 2
    REDUCE_CONV3 = 3
    REDUCE_CONV4 = 4
    REDUCE_CONV5 = 5
    REMOVE_CONV2 = 6
    REMOVE_CONV3 = 7
    REMOVE_CONV4 = 8
    QUANTIZE_INT8 = 9

ACTION_SPACE_SIZE = 10

ACTION_DESCRIPTIONS = {
    0: "Do Nothing / Terminate Episode",
    1: "Reduce Conv1 Channels by 25%",
    2: "Reduce Conv2 Channels by 25%",
    3: "Reduce Conv3 Channels by 25%",
    4: "Reduce Conv4 Channels by 25%",
    5: "Reduce Conv5 Channels by 25%",
    6: "Bypass Conv2 Layer",
    7: "Bypass Conv3 Layer",
    8: "Bypass Conv4 Layer",
    9: "Apply INT8 Dynamic Quantization"
}

def get_action_info(action_idx: int) -> Dict[str, Any]:
    return {
        "action_id": action_idx,
        "description": ACTION_DESCRIPTIONS.get(action_idx, "Unknown Action")
    }
