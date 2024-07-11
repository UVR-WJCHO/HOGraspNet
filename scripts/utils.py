## default setting ##
split_types = list(range(5))
subject_types = list(range(100))
object_types = list(range(31))
del subject_types[0]
del object_types[0]
grasp_types = [1,2,17,18,22,30,3,4,5,19,31,10,11,26,28,16,29,23,20,25,9,24,33,7,12,13,27,14]


def check_args(arg_subject, arg_object, arg_grasp):
    try:  
        if arg_subject == "all":            
            subjects = subject_types
        else:
            if arg_subject == "small":
                subjects = [1, 21, 41, 61, 81]
            elif "-" in arg_subject:
                subjects = arg_subject.split("-")
                subjects = list(range(int(subjects[0]), int(subjects[1])+1))
            else:
                subjects = arg_subject.split(",")
                subjects = list(map(int, subjects))
    except Exception as e:
        print("wrong --subject argument format. Please check the --help")

    try:
        if arg_object == "all":
            objects = object_types
        elif "-" in arg_object:
            objects = arg_object.split("-")
            objects = list(range(int(objects[0]), int(objects[1])+1))
        else:
            objects = arg_object.split(",")
            objects = list(map(int, objects))
    except Exception as e:
        print("wrong --object argument format. Please check the --help")

    try:
        if arg_grasp == "all":
            grasps = grasp_types
        else:
            grasps = arg_grasp.split(",")
            grasps = list(map(int, grasps))
    except Exception as e:
        print("wrong --grasp argument format. Please check the --help")
    
    return subjects, objects, grasps