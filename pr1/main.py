from PIL import Image

references = {
    "town":
        [
            {
                "up-left": (456, 277),
                "up-right": (476, 277),
                "down-left": (456, 292),
                "down-right": (476, 292)
            },
            {
                "up-left": (605, 231),
                "up-right": (638, 231),
                "down-left": (605, 258),
                "down-right": (538, 258)
            }
        ],
    "sea":
        [
            {
                "up-left": (623, 446),
                "up-right": (701, 446),
                "down-left": (623, 511),
                "down-right": (701, 511)
            },
            {
                "up-left": (735, 588),
                "up-right": (827, 588),
                "down-left": (735, 666),
                "down-right": (827, 666)
            }
        ],
    "forest":
        [
            {
                "up-left": (538, 108),
                "up-right": (581, 108),
                "down-left": (538, 143),
                "down-right": (581, 143)
            }
        ]
}

with Image.open("EO_Browser_images/2025-03-03-00_00_2025-03-03-23_59_Sentinel-2_L2A_B03_(Raw).jpg") as im:
    