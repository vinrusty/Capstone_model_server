import numpy as np
import random

def run_fan_optimization(a, b, c, d, goal_consumption):
    # Initialize Q-table, states, actions, and other parameters
    states = [("Morning", "Low", "Low"), ("Morning", "Medium", "Moderate"), ("Morning", "High", "High"), ("Morning", "Zero", "Zero"), ("Afternoon", "Low", "Low"), ("Afternoon", "Medium", "Moderate"), ("Afternoon", "High", "High"), ("Afternoon", "Zero", "Zero"), ("Evening", "Low", "Low"), ("Evening", "Medium", "Moderate"), ("Evening", "High", "High"), ("Evening", "Zero", "Zero"), ("Night", "Low", "Low"), ("Night", "Medium", "Moderate"), ("Night", "High", "High"), ("Night", "Zero", "Zero")]
    actions = ["Off", "Low Speed", "Medium Speed", "High Speed"]
    Q_table = np.random.rand(len(states), len(actions)) * 0.01
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.25
    num_episodes = 10

    MAX_CONSUMPTION_CHECK = False
    FAN_CONSUMPTION = 3.011
    LIMIT = FAN_CONSUMPTION * 6
    weights = [a, b, c, d]
    values = [(0, 3), (4, 7), (8, 11), (12, 15)]
    OFFSET = 0.2
    limit_check = LIMIT - OFFSET

    consumption_values = {"Zero": 0, "Low": 0.05, "Medium": 0.1, "High": 0.2}
    reward_values = {"Zero": 0, "Low": 1, "Medium": 2, "High": 5}

    def reward_function(state):
        _, brightness, _ = state
        return -reward_values[brightness]

    def action_function(choice, action_index):
        if action_index == 0:
            return choice[0] + 3
        elif action_index == 1:
            return choice[0]
        elif action_index == 2:
            return choice[0] + 1
        elif action_index == 3:
            return choice[0] + 2
        else:
            raise ValueError("Wrong action index!")

    def transition_function_2(cons, action, weights, MAX_CONSUMPTION_CHECK):
        if (cons['Morning']) >= limit_check:
            weights[0] = 0
        if (cons['Afternoon']) >= limit_check:
            weights[1] = 0
        if (cons['Evening']) >= limit_check:
            weights[2] = 0
        if (cons['Night']) >= limit_check:
            weights[3] = 0

        total_weight = sum(weights)
        if total_weight <= 0:
            MAX_CONSUMPTION_CHECK = True
            weights = [1, 1, 1, 1]
            total_weight = sum(weights)

        normalized_weights = [weight / total_weight for weight in weights]
        chosen_value = random.choices(values, weights=normalized_weights)[0]
        final_state_index = action_function(chosen_value, action)
        return final_state_index, MAX_CONSUMPTION_CHECK

    # Q-learning algorithm
    for episode in range(num_episodes):
        if MAX_CONSUMPTION_CHECK:
            break
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
            next_state_index, MAX_CONSUMPTION_CHECK = transition_function_2(cons_values, action_index, weights, MAX_CONSUMPTION_CHECK)

            Q_table[state_index, action_index] += learning_rate * (
                reward + discount_factor * np.max(Q_table[next_state_index]) - Q_table[state_index, action_index]
            )

            total_consumption += consumption_values[state[1]]
            cons_values[state[0]] += consumption_values[state[1]]

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
        next_state_index, MAX_CONSUMPTION_CHECK = transition_function_2(final_cons_values, action_index, weights, MAX_CONSUMPTION_CHECK)

        Q_table[state_index, action_index] += learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state_index]) - Q_table[state_index, action_index]
        )

        total_consumption += consumption_values[state[1]]
        final_cons_values[state[0]] += consumption_values[state[1]]

        state_index = next_state_index
        if total_consumption >= goal_consumption:
            break

    if MAX_CONSUMPTION_CHECK:
        x = LIMIT
        final_cons_values = {"Morning": x, "Afternoon": x, "Evening": x, "Night": x}

    final_cons_hours = {"Morning": 0, "Afternoon": 0, "Evening": 0, "Night": 0}
    sum_of_consumption = 0
    for i in final_cons_values.items():
        j = i[1] / FAN_CONSUMPTION
        sum_of_consumption += j
        final_cons_hours[i[0]] = j

    def redistribute_buckets(buckets):
        excess = 0
        for key, value in buckets.items():
            if value > 6:
                excess += value - 6
                buckets[key] = 6
        num_buckets = len(buckets) - sum(1 for value in buckets.values() if value >= 6)
        if num_buckets > 0:
            excess_per_bucket = excess / num_buckets
            for key, value in buckets.items():
                if value < 6:
                    buckets[key] += excess_per_bucket
        return buckets

    buckets = final_cons_hours
    buckets = redistribute_buckets(buckets)

    def decimal_to_hours_minutes(dictionary):
        result = {}
        for key, value in dictionary.items():
            hours = int(value)
            minutes = int((value - hours) * 60)
            result[key] = (hours, minutes)
        return result

    FINAL_OUTPUT_VALUES = decimal_to_hours_minutes(buckets)
    return FINAL_OUTPUT_VALUES