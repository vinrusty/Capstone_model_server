import numpy as np
import random

def run_rl_transport_4wheeler(weights, goal_consumption, fuel_type, traffic_probability):
    a, b, c, d = weights

    states = [("Morning", "1", "High"),
              ("Morning", "3", "Moderate"),
              ("Morning", "5", "Low"),
              ("Morning", "Zero", "Zero"),
              ("Morning", "1", "Moderate"),
              ("Morning", "5", "High"),
              ("Afternoon", "1", "High"),
              ("Afternoon", "3", "Moderate"),
              ("Afternoon", "5", "Low"),
              ("Afternoon", "Zero", "Zero"),
              ("Afternoon", "1", "Moderate"),
              ("Afternoon", "5", "High"),
              ("Evening", "1", "High"),
              ("Evening", "3", "Moderate"),
              ("Evening", "5", "Low"),
              ("Evening", "Zero", "Zero"),
              ("Evening", "1", "Moderate"),
              ("Evening", "5", "High"),
              ("Night", "1", "High"),
              ("Night", "3", "Moderate"),
              ("Night", "5", "Low"),
              ("Night", "Zero", "Zero"),
              ("Night", "1", "Moderate"),
              ("Night", "5", "High"),]

    actions = ["Off", "Accelerate", "Cruising", "Speeding", "Engine Braking"]

    cons_values = {"Morning": 0, "Afternoon": 0, "Evening": 0, "Night": 0}

    Q_table = np.random.rand(len(states), len(actions)) * 0.01

    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.25
    num_episodes = 100

    if fuel_type == "Petrol":
        # FOURWHEELER_CONSUMPTION = (160381 / 10000000) * 1000
        FOURWHEELER_CONSUMPTION = (6567.81 / 100000) * 1000
    elif fuel_type == "Diesel":
        # FOURWHEELER_CONSUMPTION = (1371670 / 100000000) * 1000
        FOURWHEELER_CONSUMPTION = (6335.67 / 100000) * 1000
    else:
        raise ValueError("WRONG FUEL TYPE INPUT")

    print(FOURWHEELER_CONSUMPTION)
    values = [(0, 5), (6, 11), (12, 17), (18, 23)]

    if len(weights) != len(values):
        raise ValueError("Length of weights and values should be the same.")

    consumption_values = {
        ("Zero", "Zero"): 0,
        ("1", "High"): 10,
        ("3", "Moderate"): 5,
        ("5", "Low"): 2,
        ("1", "Moderate"): 7,
        ("5", "High"): 8
    }
    reward_values = {
        ("Zero", "Zero"): 0,
        ("1", "High"): 10,
        ("3", "Moderate"): 5,
        ("5", "Low"): 2,
        ("1", "Moderate"): 7,
        ("5", "High"): 8
    }

    def reward_function(state):
        _, gear, consumption = state
        return -reward_values[(gear, consumption)]

    def action_function(choice, action_index):
        if np.random.uniform(0, 1) < traffic_probability:
            return choice[0]
        if action_index == 0:
            return choice[0] + 3
        elif action_index == 1:
            return choice[0] + 1
        elif action_index == 2:
            return choice[0] + 2
        elif action_index == 3:
            return choice[0] + 5
        elif action_index == 4:
            a = np.random.randint(0, 2)
            if a == 1:
                return choice[0] + 4
            elif a == 2:
                return choice[0] + 1
            else:
                return choice[0]
        else:
            raise ValueError("Wrong action index!")

    def transition_function_2(cons, action, weights):
        total_weight = sum(weights)
        normalized_weights = [weight / total_weight for weight in weights]
        chosen_value = random.choices(values, weights=normalized_weights)[0]
        final_state_index = action_function(chosen_value, action)
        return final_state_index

    for episode in range(num_episodes):
        weights = [a, b, c, d]
        total_consumption = 0
        state_index = random.randint(0, 3)
        cons_values = {"Morning": 0, "Afternoon": 0, "Evening": 0, "Night": 0}

        while True:
            if np.random.uniform(0, 1) < epsilon:
                action_index = np.random.randint(len(actions))
            else:
                action_index = np.argmax(Q_table[state_index])

            action = actions[action_index]
            state = states[state_index]

            reward = reward_function(state)
            next_state_index = transition_function_2(cons_values, action_index, weights)

            Q_table[state_index, action_index] += learning_rate * (
                    reward + discount_factor * np.max(Q_table[next_state_index]) - Q_table[state_index, action_index])

            total_consumption += consumption_values[(state[1], state[2])]
            cons_values[state[0]] += consumption_values[(state[1], state[2])]

            state_index = next_state_index

            if total_consumption >= goal_consumption:
                break

    final_cons_values = {"Morning": 0, "Afternoon": 0, "Evening": 0, "Night": 0}

    weights = [a, b, c, d]
    total_consumption = 0
    state_index = random.randint(0, 3)

    while True:
        if np.random.uniform(0, 1) < epsilon:
            action_index = np.random.randint(len(actions))
        else:
            action_index = np.argmax(Q_table[state_index])

        action = actions[action_index]
        state = states[state_index]

        reward = reward_function(state)
        next_state_index = transition_function_2(final_cons_values, action_index, weights)

        Q_table[state_index, action_index] += learning_rate * (
                reward + discount_factor * np.max(Q_table[next_state_index]) - Q_table[state_index, action_index])

        total_consumption += consumption_values[(state[1], state[2])]

        final_cons_values[state[0]] += consumption_values[(state[1], state[2])]

        state_index = next_state_index
        if total_consumption >= goal_consumption:
            print(f"Goal consumption reached in episode {episode + 1} with consumption {total_consumption}.")
            break

    final_cons_hours = {"Morning": 0, "Afternoon": 0, "Evening": 0, "Night": 0}
    sum_of_consumption = 0
    for i in final_cons_values.items():
        j = i[1] / FOURWHEELER_CONSUMPTION
        sum_of_consumption += j
        final_cons_hours[i[0]] = j
    print("Number of Kilometers")
    print(final_cons_hours)
    print("Goal consumption Kilometers " + str(goal_consumption / FOURWHEELER_CONSUMPTION))
    print("Achieved Sum: " + str(sum_of_consumption))

    def round_dict_values(dictionary):
        rounded_dict = {}
        for key, value in dictionary.items():
            rounded_dict[key] = round(value, 2)
        return rounded_dict

    rounded_dict = round_dict_values(final_cons_hours)

    FINAL_OUTPUT_VALUES = rounded_dict
    print("Number of Kilometers:")
    print(rounded_dict)

    return FINAL_OUTPUT_VALUES