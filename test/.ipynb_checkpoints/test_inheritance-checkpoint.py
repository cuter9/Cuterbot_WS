from jetbot import ObjectFollower, RoadCruiserTRT, FleeterTRT

'''
class FleeterTRT_1(RoadCruiserTRT, ObjectFollower):
    # ObjectFollower.__init__(self, init_sensor=False)
    # RoadCruiserTRT.__init__(self, init_sensor=False)
    def __init__(self, init_sensor=False):
        # super().__init__()
        RoadCruiserTRT.__init__(self, init_sensor_rc=init_sensor)
        ObjectFollower.__init__(self, init_sensor_of=init_sensor)
        self.a = 1
'''


class FleeterTRT_1(ObjectFollower, RoadCruiserTRT):
    # ObjectFollower.__init__(init_sensor=False)
    # RoadCruiserTRT.__init__(init_sensor=False)
    def __init__(self, init_sensor=True):
        # super().__init__()
        RoadCruiserTRT.__init__(self, init_sensor_rc=False)
        ObjectFollower.__init__(self, init_sensor_of=False)
        self.a = 1


of = FleeterTRT_1()
of.a = 2

