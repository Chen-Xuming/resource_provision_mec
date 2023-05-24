from codes.min_cost_v4.env.service import Service

class User:
    def __init__(self, user_id):
        self.user_id = user_id

        self.arrival_rate = 0

        self.service_A = None   # type: Service
        self.service_B = None   # type: Service
        self.service_R = None   # type: Service

    def reset(self):
        self.service_A.reset()
        self.service_R.reset()