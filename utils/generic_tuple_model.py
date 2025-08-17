class TupleModel(GenericModel):
    def __init__(self, state_dim, action_dim) -> None:
        self.state_dim = 1
        self.action_dim = 1
        super().__init__(state_dim, action_dim)

    def build_state_space(self, tuple_length: int, tuple_min: tuple, tuple_max: tuple):
        for tuple_couple in range(tuple_length):
            self._update_state_space(
                tuple_min[tuple_couple],
                tuple_max[tuple_couple],
                bool(tuple_couple == 0),
            )

        self.state_space = {
            state: state_index for (state_index, state) in enumerate(self.state_list)
        }

    def _update_state_space(
        self, tuple_min: int, tuple_max: int, first_tuple: bool = False
    ):
        if first_tuple:
            self.state_list = [
                (tuple_elem,) for tuple_elem in range(tuple_min, tuple_max + 1)
            ]
        else:
            new_state_list = []
            for tuple_elem in range(tuple_min, tuple_max + 1):
                for state in self.state_list:
                    new_state = state + (tuple_elem,)
                    new_state_list.append(new_state)
            self.state_list = new_state_list

        self.state_dim *= tuple_max - tuple_min + 1

    def build_action_space(self, tuple_length: int, tuple_min: tuple, tuple_max: tuple):
        for tuple_couple in range(tuple_length):
            self._update_action_space(
                tuple_min[tuple_couple],
                tuple_max[tuple_couple],
                bool(tuple_couple == 0),
            )

        self.action_space = {
            action: action_index
            for (action_index, action) in enumerate(self.action_list)
        }

    def _update_action_space(
        self, tuple_min: int, tuple_max: int, first_tuple: bool = False
    ):
        if first_tuple:
            self.action_list = [
                (tuple_elem,) for tuple_elem in range(tuple_min, tuple_max + 1)
            ]
        else:
            new_action_list = []
            for tuple_elem in range(tuple_min, tuple_max + 1):
                for action in self.action_list:
                    new_action = action + (tuple_elem,)
                    new_action_list.append(new_action)
            self.state_list = new_action_list

        self.state_dim *= tuple_max - tuple_min + 1