from enum import Enum

## default dataset types ##
split_types = list(range(5))
subject_types = list(range(100))
object_types = list(range(31))
del subject_types[0]
del object_types[0]
grasp_types = [1,2,17,18,22,30,3,4,5,19,31,10,11,26,28,16,29,23,20,25,9,24,33,7,12,13,27,14]

## parameters ##
base_url_set = ["images_augmented", "annotations", "extra", "images"]

_CAMIDSET = ['mas', 'sub1', 'sub2', 'sub3']

_TEST_OBJ_LIST = ['10','02','12','20','04','06','11']
_TEST_GRASP_LIST = ['23','25','29','16']

class OBJType(IntEnum):
    cracker_box = 1
    potted_meat_can = 2
    banana = 3
    apple = 4
    wine_glass = 5
    bowl = 6
    mug = 7
    plate = 8
    spoon = 9
    knife = 10
    small_marker = 11
    spatula = 12
    flat_screwdriver = 13
    hammer = 14
    baseball = 15
    golf_ball = 16
    credit_card = 17
    dice = 18
    disk_lid = 19
    smartphone = 20
    mouse = 21
    tape = 22
    master_chef_can = 23
    Scrub_cleanser_bottle = 24
    large_marker = 25
    stapler = 26
    note = 27
    scissors = 28
    foldable_phone = 29
    cardboard_box = 30


CFG_OBJECT_SCALE_FIXED = [1.,
1.,
1.,
1.,
0.8296698468,
1.,
1.,
1.,
1.,
1.,
0.1035083229,
1.,
0.6706711338,
1.,
1.,
0.43,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.]