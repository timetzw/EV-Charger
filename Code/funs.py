import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import config


# np.random.seed(2018)

# function of making choices
def make_new_choice(slots, data, E0):
    m = slots
    eps = 0.0001  # Used to avoid zero at denominator
    alpha = config.alpha  # have to be negative because the lower the price the better
    price = data.iloc[:, 0:m]
    price = pd.DataFrame(price)
    price.reset_index()
    price = price.add(eps)
    utl_price = (price) ** alpha

    beta = config.beta
    num = data.iloc[:, m : 2 * m]
    num = pd.DataFrame(num)
    num.reset_index()
    num = num.add(eps)
    # for iter_1 in range(num.shape[0]):
    #     for iter_2 in range(num.shape[1]):
    #         expected_ratio = E0.iloc[iter_1, iter_2]
    #         if expected_ratio >= 0.95: expected_ratio -= 0.05
    #         num.iloc[iter_1, iter_2] = max(
    #             round(num.iloc[iter_1, iter_2] * (1 - expected_ratio)), 0 + eps
    #         )
    utl_num = (num) ** beta

    epsi = 0.1
    gamma = (
        config.gamma
    )  # have to be negative because the shorter the distance the better
    dist = data.iloc[:, 2 * m : 3 * m]
    dist = pd.DataFrame(dist)
    utl_dist = (dist.add(epsi)) ** gamma

    utl_total = pd.DataFrame(utl_price.values * utl_num.values * utl_dist.values)
    # choice = pd.DataFrame(columns=utl_total.columns, index=utl_total.index)
    choice = utl_total.idxmax(axis=1)
    for i, row in utl_total.iterrows():
        # Compare the values in each column
        if row[0] > row[1]:
            choice.iloc[i] = 0
        elif row[1] > row[0]:
            choice.iloc[i] = 1
        else:
            # Toss a coin if the values are equal
            choice.iloc[i] = np.random.binomial(1, 0.5)
    # utl_total = pd.DataFrame(utl_price.values * utl_dist.values)
    # Adjusted Choice_Value
    utl_dist = []
    utl_price = []
    j = 0
    for i in choice:
        utl_dist.append((dist.iloc[j, i] + epsi) ** gamma)
        utl_price.append((price.iloc[j, i]) ** alpha)
        j += 1
    utl_dist = pd.DataFrame(utl_dist)
    utl_price = pd.DataFrame(utl_price)
    utl_total_modified = pd.DataFrame(utl_price.values * utl_dist.values)

    choice_value = utl_total.max(axis=1)
    data.iloc[:, -1] = choice
    print(sum(choice_value))
    return data, choice_value


# m: the number of drivers
def generate_first_data(
    slots, m, max_time, scenario, seed, price_ctl, dist_ctl, arrive_ctl
):
    # Scenario: 0 represents normal, i(i>=1) represents game day which happens close to the (i-1)-th parking lot.

    # The price of each parking slot
    # np.random.randint is discrete uniformly distributed
    if price_ctl == 0:  # not random
        if scenario == 0:
            # VARIOUS PRICE Only consider n=slots parking lots, and only one price will change due to the GAME. If
            # only two parking lots, then here we specify the number price specifically to be "price_base" and
            # "price_change"
            if slots == 2:
                price_store = [np.random.randint(
                    price_base, price_base + 1), np.random.randint(
                    price_change, price_change + 1)]
            else:
                price_store = np.random.randint(
                    price_base, price_base + 1, size=(slots)
                )
            price_store = [price_store for _ in range(m)]
            price_arr = pd.DataFrame.from_records(price_store)

        else:

            if slots == 2:
                price_store = [np.random.randint(
                    price_base, price_base + 1), np.random.randint(
                    price_change, price_change + 1)]
            else:
                price_store = np.random.randint(
                    price_base, price_base + 1, size=(slots)
                )
            price_store[scenario - 1] = price_store[scenario - 1] * price_gameday_multiple
            price_store = [price_store for _ in range(m)]
            price_arr = pd.DataFrame.from_records(price_store)
    else:  # random
        np.random.seed(config.seed)
        if scenario == 0:
            # SAME PRICE
            # price = np.random.randint(
            #     5, 30, size=slots
            # )  # generate random price for each parking slot
            if slots == 2:
                price_store = [np.random.randint(
                    max(price_base - 3, 1), price_base + 3), np.random.randint(
                    max(price_change - 3, 1), price_change + 3)]
            else:
                price_store = np.random.randint(
                    max(price_base - 3, 1), price_base + 3, size=(slots)
                )
            price_store = [price_store for _ in range(m)]
            price_arr = pd.DataFrame.from_records(price_store)

        else:
            # There are some game at "scenario" lots.
            # For experiment, if we only consider two parking lots, then it will be the first.
            if slots == 2:
                price_store = [np.random.randint(
                    max(price_base - 3, 1), price_base + 3), np.random.randint(
                    max(price_change - 3, 1), price_change + 3)]
            else:
                price_store = np.random.randint(
                    max(price_base - 3, 1), price_base + 3, size=(slots)
                )
            price_store[scenario - 1] = price_store[scenario - 1] * price_gameday_multiple
            price_store = [price_store for _ in range(m)]
            price_arr = pd.DataFrame.from_records(price_store)

    begin_arr = pd.DataFrame(price_arr)

    # Create the initial distribution of chargers
    as_arr = []
    for i in range(slots):
        as_arr.append(np.ones(m) * 10)
    as_arr = pd.DataFrame(as_arr)
    as_arr = as_arr.T

    # Create the VALUE of Parking Lots, which is the distance from the lot to final destination; for each driver, the
    # value may very

    dist_arr = []
    if dist_ctl == 0:  # not random
        if scenario == 0:
            if slots == 2:
                dist_store = [[np.random.randint(dist_base,dist_base+1), np.random.randint(dist_change,dist_change+1)] for _ in range(m)]
                dist_arr = pd.DataFrame.from_records(dist_store)
            else:
                dist_store = [np.random.randint(dist_base,dist_base+1, size=slots) for _ in range(m)]
                dist_arr = pd.DataFrame.from_records(dist_store)
        else:
            if slots == 2:
                base = dist_base
                change = dist_change
                if scenario == 1:
                    base = config.dist_base * dist_gameday_multiple
                if scenario == 2:
                    change = config.dist_change * dist_gameday_multiple
                dist_store = [[np.random.randint(base,base+1), np.random.randint(change,change+1)] for _ in range(m)]
                dist_arr = pd.DataFrame.from_records(dist_store)
            else:
                dist_store = [np.random.randint(dist_base,dist_base+1, size=slots) for _ in range(m)]
                dist_arr = pd.DataFrame.from_records(dist_store)
                dist_arr[scenario - 1] = (dist_arr[scenario - 1] * dist_gameday_multiple).round(0).astype(int)
    else:
        np.random.seed(dist_ctl)
        if scenario == 0:
            if slots == 2:
                dist_store = [[np.random.randint(dist_base*0.7,dist_base), np.random.randint(dist_change*0.7,dist_change)] for _ in range(m)]
                dist_arr = pd.DataFrame.from_records(dist_store)
            else:
                dist_store = [np.random.randint(dist_base*0.7, dist_base, size=slots) for _ in range(m)]
                dist_arr = pd.DataFrame.from_records(dist_store)

        else:
            if slots == 2:
                dist_store = [[np.random.randint(dist_base*0.7,dist_base), np.random.randint(dist_change*0.7,dist_change)] for _ in range(m)]
                dist_arr = pd.DataFrame.from_records(dist_store)
            else:
                dist_store = [np.random.randint(dist_base*0.7, dist_base, size=slots) for _ in range(m)]
                dist_arr = pd.DataFrame.from_records(dist_store)
                dist_arr[scenario - 1] = (dist_arr[scenario - 1] * dist_gameday_multiple).round(0).astype(int)

    # the estimated distance between drivers and parking lots, measured in time
    # Now only works for 2 Parking lots scenario
    arrive_arr = []
    if arrive_ctl == 0:  # no random, ascending
        if scenario == 0:
            for i in range(slots):
                temp = np.ceil(
                    np.linspace(1 - 1, config.num_cars - 1, config.num_cars)
                    * config.observation_time
                    / config.num_cars
                )
                for j in range(len(temp)):
                    if temp[j] >= config.observation_time - 1:
                        temp[j] = config.observation_time - 1
                arrive_arr.append(temp)
            arrive_arr = pd.DataFrame(arrive_arr)
            arrive_arr = arrive_arr.T
        else:
            for i in range(slots):
                temp = np.ceil(
                    np.linspace(1 - 1, config.num_cars - 1, config.num_cars)
                    * config.observation_time
                    / config.num_cars
                )
                for j in range(len(temp)):
                    if temp[j] >= config.observation_time - 1:
                        temp[j] = config.observation_time - 1
                # Game Day Let 80% people come in within 10% of time
                for j in range(int(0.2 * len(temp)), int(0.8 * len(temp))):
                    temp[j] = np.random.randint(
                        0.45 * config.observation_time, 0.55 * config.observation_time
                    )
                arrive_arr.append(temp)
            arrive_arr = pd.DataFrame(arrive_arr)
            arrive_arr = arrive_arr.T
    else:
        np.random.seed(config.seed)
        if scenario == 0:  # Non Game Day
            for i in range(slots):
                arrive_arr.append(np.random.randint(0, max_time, size=m))
            arrive_arr = pd.DataFrame(arrive_arr)
            arrive_arr = arrive_arr.T
        else:  # Game Day
            for i in range(slots):
                temp = np.random.randint(0, max_time, size=m)
                for j in range(int(0.2 * len(temp)), int(0.8 * len(temp))):
                    temp[j] = np.random.randint(
                        0.45 * config.observation_time, 0.55 * config.observation_time
                    )
                arrive_arr.append(temp)
            arrive_arr = pd.DataFrame(arrive_arr)
            arrive_arr = arrive_arr.T

    # Create initial decisions
    d_arr = []
    d_arr = dist_arr.idxmax(axis=1)

    # Combine to one DataFrame
    df = []
    df = pd.concat([begin_arr, as_arr, dist_arr, arrive_arr, d_arr], axis=1)

    # Name of Columns
    dfindex = []
    k = 4  # number of different type of data
    for i in range(slots * k + 1):
        if i < slots:
            dfindex.append("The charging price of [{num}]".format(num=i))
        elif i < 2 * slots:
            dfindex.append("Chargers in [{num}]".format(num=i - slots))
        elif i < 3 * slots:
            dfindex.append(
                "Distance to final destination from [{num}]".format(num=i - 2 * slots)
            )
        elif i < 4 * slots:
            dfindex.append(
                "Distance from current place to [{num}]".format(num=i - 3 * slots)
            )
        else:
            dfindex.append("Choice")
    df.columns = dfindex

    return df


def inflow(df, max_time, scenario):
    slots_num = int((df.shape[1] - 1) / 4)
    m = int(df.shape[0])  # number of drivers
    # generate inflow data
    inflow = pd.DataFrame(np.zeros((max_time, slots_num)))
    for i in range(m):
        choice = int(df.iloc[i, -1])
        time = int(
            df.iloc[i, choice + 3 * slots_num]
        )  # Here 3*slots~(3*slots+m) is distance from current place to pl.
        if time < max_time:
            inflow.iloc[time, choice] += 1
    pd.DataFrame(inflow).to_csv(f"./data/input_Sce{scenario}.csv".format(scenario))
    return inflow


def show(max):
    x_cord = ["station-" + str(i) for i in range(len(max))]
    plt.bar(x_cord, max[:, 1])
    plt.ylabel("The number of allocated chargers")
    plt.xlabel("Charge station")
    plt.title("Visualization of the model solution")
    plt.show()


def update(df, max):
    # new_slots = np.array(max[:, 1])
    new_slots = np.array(max)
    df_new = df
    for i in range(np.shape(df_new)[0]):
        df_new.iloc[i, slots : 2 * slots] = new_slots
    df = df_new
    return df


def res_mod(pred):
    result = pred.detach().numpy()
    result = result.round()
    for i in range(len(result)):
        if result[i] <= 0:
            result[i] = 0
    return result


def stay_or_not(waiting):
    before = sum(waiting)
    leave_prob = config.leave_prob
    for i in range(len(waiting)):
        n = int(waiting[i])
        p = max(0, min(1, leave_prob * i))  #The probability of leaving the system
        decision = np.random.binomial(n, p)
        waiting[i] -= decision
    after = sum(waiting)
    rejected = before - after
    return waiting, rejected


def time_order_df(df, slots):
    temp_df = df
    column_name = temp_df.columns
    sort_reference = [column_name[-i - 1] for i in range(slots, 0, -1)]
    sort_reference.insert(0, column_name[-1])
    temp_df = temp_df.sort_values(by=sort_reference[0])
    temp = temp_df.groupby("Choice")
    group_keys = list(temp.groups.keys())
    temp = temp_df.groupby(["Choice"])
    temporary_df_list = []
    for i in group_keys:
        temporary_df = temp.get_group(i)
        temporary_df_list.append(temporary_df.sort_values(by=sort_reference[i + 1]))
    temp_df = pd.concat(temporary_df_list, axis=0)
    return temp_df


def simulation(df, inflow, allocation, choice_value, leave_control):
    global max_time

    num_of_pl = inflow.shape[1]
    occupied_df = pd.DataFrame(np.zeros([max_time, num_of_pl]))
    rejection = pd.DataFrame(np.zeros([max_time, num_of_pl]))
    entry_df = pd.DataFrame(np.zeros([max_time, num_of_pl]))
    charging_time_base = config.charging_time_base
    now_in_queue_df = pd.DataFrame(np.zeros([max_time, num_of_pl]))
    preference = np.zeros(df.shape[0])
    # Preference value setting
    # Setting 1: The relative proportion preference
    # pref_df = df.iloc[:, 2 * num_of_pl + 0:2 * num_of_pl + num_of_pl]
    # for i in range(pref_df.shape[0]):
    #     total = sum(pref_df.iloc[i, :])
    #     for j in range(num_of_pl):
    #         pref_df.iloc[i, j] = pref_df.iloc[i, j] / total
    # for i in range(df.shape[0]):
    #     preference[i] = pref_df.iloc[i, df.iloc[i, -1]]

    # Setting 2: the choice value
    preference = np.array(choice_value.copy())

    go_or_not = []

    for i in range(num_of_pl):
        come_in = np.array(inflow.iloc[:, i].T)  # i-th pl inflow, stable
        entry = np.zeros(max_time)  # new entry
        depart = np.zeros(max_time)  # new departure
        occupied = np.zeros(max_time)  # new occupation
        waiting = np.array([])  # waiting queue, dynamic
        now_in_queue = np.zeros(max_time)
        storage = allocation[i]  # the maximum store
        overflow = 0  # Record the number of unfinished cars
        leave_record = np.zeros(max_time)  # Record the number of leaving cars at time k

        for j in range(max_time):
            waiting = np.insert(waiting, 0, come_in[j])
            # entering process
            occupied[j] -= depart[j]
            now_in_queue[j] = np.sum(waiting)
            # entry update
            if occupied[j] < storage:
                num_allow_to_enter = int(min(storage - occupied[j], now_in_queue[j]))
                if num_allow_to_enter > 0:
                    if charging_time_switch == 1:
                        charging_time = np.rint(
                            np.random.uniform(
                                -config.charging_time_range / 2,
                                config.charging_time_range / 2,
                                num_allow_to_enter,
                            )
                            + charging_time_base
                        )
                    else:
                        charging_time = np.array(
                            [charging_time_base for _ in range(num_allow_to_enter)]
                        )
                    for ite in range(num_allow_to_enter):
                        leave_idx = j + int(charging_time[ite])
                        if leave_idx <= max_time - 1:
                            depart[leave_idx] += 1
                        else:
                            overflow += 1
                entry[j] = num_allow_to_enter
                occupied[j] = occupied[j] + num_allow_to_enter
                now_in_queue[j] -= entry[j]
            else:
                num_allow_to_enter = 0
                entry[j] = num_allow_to_enter
            if j != max_time - 1:
                occupied[j + 1] = occupied[j]
            # probability decision
            left = num_allow_to_enter
            pointer = len(waiting) - 1
            while left > 0:
                if waiting[pointer] == 0:
                    pointer -= 1
                else:
                    deduction = min(waiting[pointer], left)
                    waiting[pointer] -= deduction
                    left -= deduction
                    pointer -= 1
            waiting_before = waiting.copy()
            waiting, rejection.iloc[j, i] = stay_or_not(waiting)
            waiting_after = waiting.copy()
            waiting_change = waiting_before - waiting_after
            for k in range(len(waiting_change) - 1, -1, -1):
                leave_record[len(waiting_change) - 1 - k] += waiting_change[k]
            # Add those overflow cars into the leaving record
            if j == max_time - 1:
                for k in range(len(waiting_after) - 1, -1, -1):
                    leave_record[len(waiting_after) - 1 - k] += waiting_after[k]
            now_in_queue[j] -= rejection.iloc[j, i]
        entry_df.iloc[:, i] = entry.T
        occupied_df.iloc[
            :, i
        ] = occupied.T  # Store the occupation situation of i-th parking lot
        rejected = [sum(rejection.iloc[:, i]) for i in range(num_of_pl)]
        now_in_queue_df.iloc[:, i] = now_in_queue.T

        all_cars = int(sum(come_in))
        temp = np.ones(all_cars)
        for k in range(max_time):
            if (
                leave_record[k] != 0
            ):  # if at time k there are some cars leaving the system
                current = int(
                    max(sum(come_in[:k]) - 1, 0)
                )  # @current: the position of the car queue at time k
                for j in range(int(leave_record[k])):
                    temp[current - j] = 0

        go_or_not.append(temp)
    go_or_not = np.concatenate(
        go_or_not
    )  # concat parallelly, w.r.t the serial number of parking lot
    if leave_control == 0:
        preference = preference
    else:
        preference = preference * go_or_not
    # preference = preference
    print(
        "allo = {}, as 0 = {}, as 1 = {}".format(
            allocation, len(df[df["Choice"] == 0]), len(df[df["Choice"] == 1])
        )
    )
    # choice_num = len(df[df['Choice'] == 0])/(len(df[df['Choice'] == 0])+len(df[df['Choice'] == 1]))
    choice_num = len(df[df["Choice"] == 0]) / df.shape[0]
    driver_choice = len(df[df["Choice"] == 0])
    return (
        rejected,
        occupied_df,
        rejection,
        entry_df,
        now_in_queue_df,
        preference,
        choice_num,
        driver_choice
    )


class report:
    def __init__(self, df, file, scenario, seed):
        # np.random.seed(seed)
        self.min_total_rejected = np.inf
        # self.min_total_target = np.inf
        self.max_total_target = -np.inf
        self.max_inflow = []
        self.max_rejected = []
        self.max_allocation = []
        self.max_pref_reward = []
        self.df = df
        self.file = file
        self.scenario = scenario
        self.theta = config.theta
        self.occupied = []
        self.rejection = []
        self.entry = []
        self.now_in_queue = []
        self.weighted_efficiency = []
        self.efficiency = []
        self.all_pref = []
        self.weighted_preference = []
        self.best_df = []
        self.pref_weight = []
        self.efficiency_weight = []
        self.preference = []
        self.choice_value = []
        self.pref_store = []
        self.eff_store = []
        self.target_store = []
        self.choice_num_store = []
        self.driver_choice_store = []
    def run(self, result):
        global slots
        # Step1: Update the data matrix with new allocation
        self.df = update(self.df, result)
        E0 = self.E0()
        self.df, choice_value = make_new_choice(slots, self.df, E0)
        self.df = time_order_df(self.df, slots)
        self.df = self.df.reset_index(drop=True)
        temp_df = self.df.copy()
        entry_flow = inflow(self.df, max_time, self.scenario)
        (
            rejected,
            occupied,
            rejection,
            entry,
            now_in_queue,
            preference,
            choice_num,
            driver_choice
        ) = simulation(temp_df, entry_flow, result, choice_value, config.leave_control)
        total_rejected = sum(rejected)
        new_choice = self.df.iloc[:, -1]

        # OBJ as Efficiency
        efficiency = np.array(
            [
                sum(occupied.iloc[:, i])
                / (max(result[i], 0.001) * config.observation_time)
                for i in range(slots)
            ]
        )
        efficiency_weight = np.array(result / sum(result))
        weighted_efficiency = efficiency * efficiency_weight
        # preference design
        # pref_weight = np.array([sum(entry.iloc[:, i]) / max(0.001, (sum(entry_flow.iloc[:, i]))) for i in range(slots)])
        # all_pref = np.array([0.0 for _ in range(slots)])
        # choice = np.array([0 for _ in range(slots)])
        # for i in range(m):
        #     for j in range(slots):
        #         if new_choice.iloc[i] == j:
        #             choice[j] += 1
        #             all_pref[j] = all_pref[j] + self.df.iloc[i, 3 * slots + new_choice.iloc[i]] / (
        #                 sum(self.df.iloc[i, 3 * slots + k] for k in range(slots)))
        # temp_choice = np.zeros(slots)
        # for j in range(slots):
        #     if choice[j] == 0:
        #         temp_choice[j] = 1
        #     else:
        #         temp_choice[j] = choice[j]
        # all_pref = all_pref / temp_choice
        # weighted_preference = all_pref * pref_weight * efficiency_weight

        # theta = self.theta  # customize the ratio
        # total_target = 100 * sum(weighted_efficiency) + theta * sum(preference)
        if sum(weighted_efficiency) >= 0.75:
            total_target = sum(preference)
        else:
            total_target = 0
        if total_target > self.max_total_target + 0.0001:
            self.max_total_target = total_target
            self.max_rejected = total_rejected
            self.max_inflow = entry_flow
            self.max_allocation = result
            self.occupied = occupied
            self.rejection = rejection
            self.entry = entry
            self.now_in_queue = now_in_queue
            self.efficiency = efficiency
            self.weighted_efficiency = weighted_efficiency
            self.best_df = temp_df
            self.efficiency_weight = efficiency_weight
            self.preference = preference
            self.choice_value = choice_value
        self.result_print(
            current=True,
            target=total_target,
            allocation=result,
            total_rejected=total_rejected,
            efficiency=efficiency,
            weighted_efficiency=weighted_efficiency,
            sum_eff=sum(weighted_efficiency),
            preference=sum(preference),
        )
        # self.result_write(current=True, target=total_target, allocation=result, total_rejected=total_rejected,
        #                   efficiency=efficiency, preference=sum(preference))
        # self.figure_display()
        print()
        self.eff_store.append(sum(weighted_efficiency))
        self.pref_store.append(sum(preference))
        self.target_store.append(total_target)
        self.choice_num_store.append(choice_num)
        self.driver_choice_store.append(driver_choice)

    # E0 is the expected occupied ratio of chargers in each parking lot
    def E0(self):
        m = int((self.df.shape[1] - 1) / 4)  # the number of parking lot
        num_slot = self.df.iloc[:, m : 2 * m]  # chargers in each parking lot
        ratio = self.df.iloc[
            :, 2 * m : 3 * m
        ].copy()  # ratio is based on the distance to destination
        for iter in range(self.df.shape[0]):
            if sum(self.df.iloc[iter, 2 * m : 3 * m]) == 0:
                ratio.iloc[iter, :] = 1  # avoid zero
            else:
                ratio.iloc[iter, :] = self.df.iloc[iter, 2 * m : 3 * m] / sum(
                    self.df.iloc[iter, 2 * m : 3 * m]
                )

        # result = round(num_slot.sum(axis=0) / num_slot.shape[0] * 0.75)

        return ratio

    def arrangement(self, allocation, location, remain):
        self.run(np.array([10, 10]))
        # global upper_bound
        # if remain == 0:
        #     self.run(allocation)
        #     return
        # if location == len(allocation + 1):
        #     return
        # limit = upper_bound + 1 if (remain + 1) > (upper_bound + 1) else remain + 1
        # for i in range(limit):
        #     result_a = copy.deepcopy(allocation)
        #     result_a[location] = i
        #     self.arrangement(result_a, location + 1, remain - i)

    def expected_minimum_rejection(self):
        total_cars = config.num_cars
        charge_time = config.charging_time_base
        total_time = config.observation_time
        loop_times = total_time // charge_time
        chargers = config.total_chargers
        total_served = loop_times * chargers
        minimum_rejection = total_cars - total_served
        return minimum_rejection

    def result_print(self, current, target, allocation, **kwargs):
        # print('Optimal: ', end='\t')
        # print('Target: {:.3f}'.format(self.max_total_target), end='\t')
        np.set_printoptions(
            formatter={"float_kind": "{:.3f}".format}
        )  # set output format as 3 digits
        # print('Allocation: {}'.format(self.max_allocation), end='\t')
        #
        # for key, value in kwargs.items():
        #     if key == 'total_rejected':
        #         print('{}: {}'.format(key, self.max_rejected), end='\t')
        #     elif key == 'efficiency':
        #         print('{}: {}'.format(key, self.efficiency), end='\t')
        #     elif key == 'weighted_efficiency':
        #         print('{}: {}'.format(key, self.weighted_efficiency), end='\t')
        #     elif key == 'all_pref':
        #         print('{}: {}'.format(key, self.all_pref), end='\t')
        #     elif key == 'weighted_preference':
        #         print('{}: {}'.format(key, self.weighted_preference), end='\t')
        #     elif key == 'preference':
        #         print('{}: {:.3f}'.format(key, sum(self.preference)), end='\t')
        #     elif key == 'sum_eff':
        #         print('{}: {:.3f}'.format(key, sum(self.weighted_efficiency)), end='\t')
        # print()
        if current:
            print("Current:", end="\t")
            print("Target: {:.3f}".format(target), end="\t")
            print("Allocation: {}".format(allocation), end="\t")
            for key, value in kwargs.items():
                if key == "preference":
                    print("{}: {:.3f}".format(key, value), end="\t")
                elif key == "sum_eff":
                    print("{}: {:.3f}".format(key, value), end="\t")
                else:
                    print("{}: {}".format(key, value), end="\t")
            print()

    def result_write(self, current, target, allocation, **kwargs):
        self.file.write("Optimal: ", end="\t")
        self.file.write("Target: {:.3f}".format(self.max_total_target), end="\t")
        np.set_printoptions(
            formatter={"float_kind": "{:.3f}".format}
        )  # set output format as 3 digits
        self.file.write("Allocation: {}".format(self.max_allocation), end="\t")

        for key, value in kwargs.items():
            if key == "total_rejected":
                self.file.write("{}: {}".format(key, self.max_rejected), end="\t")
            elif key == "efficiency":
                self.file.write("{}: {}".format(key, self.efficiency), end="\t")
            elif key == "weighted_efficiency":
                self.file.write(
                    "{}: {}".format(key, self.weighted_efficiency), end="\t"
                )
            elif key == "all_pref":
                self.file.write("{}: {}".format(key, self.all_pref), end="\t")
            elif key == "weighted_preference":
                self.file.write(
                    "{}: {}".format(key, self.weighted_preference), end="\t"
                )
            elif key == "preference":
                self.file.write(
                    "{}: {:.3f}".format(key, sum(self.preference)), end="\t"
                )
        self.file.write()
        if current:
            self.file.write("Current:", end="\t")
            self.file.write("Target: {:.3f}".format(target), end="\t")
            self.file.write("Allocation: {}".format(allocation), end="\t")
            for key, value in kwargs.items():
                if key == "preference":
                    self.file.write("{}: {:.3f}".format(key, value), end="\t")
                else:
                    self.file.write("{}: {}".format(key, value), end="\t")
            self.file.write()

    def result_display(self):
        print(
            "target: {:.2f}, max allocation: {}, with eff:{}".format(
                self.max_total_target, self.max_allocation, self.efficiency
            )
        )

    def figure_display(self, x_stick_on):
        def annotate_axes(ax, text, fontsize=18):
            ax.text(
                0.5,
                0.5,
                text,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="darkgrey",
            )

        if slots > 2:
            x_stick_on = False

        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        gs0 = fig.add_gridspec(1, 1)

        gs00 = gs0[0].subgridspec(5, slots)
        # if x_stick_on:
        #     gs01 = gs0[1].subgridspec(4, 1)
        # else:
        #     gs01 = gs0[1].subgridspec(3, 1)
        axs00 = gs00.subplots()
        # axs01 = gs01.subplots()

        for b in range(slots):
            target = b
            idx = 0
            # Occupied
            axs00[idx, b].bar(
                np.linspace(1, max_time, max_time), self.occupied.iloc[:, target].T
            )
            axs00[idx, b].axhline(
                y=self.max_allocation[target], color="r", linestyle="-"
            )
            axs00[idx, b].title.set_text("Occupied in [{}]".format(b + 1))
            idx += 1

            # Queue Flow
            axs00[idx, b].bar(
                np.linspace(1, max_time, max_time), self.now_in_queue.iloc[:, target]
            )

            axs00[idx, b].title.set_text("Queue in [{}]".format(b + 1))
            idx += 1

            # Real Entry Flow
            axs00[idx, b].bar(
                np.linspace(1, max_time, max_time), self.entry.iloc[:, target]
            )
            axs00[idx, b].title.set_text("Entry in [{}]".format(b + 1))
            idx += 1

            # Rejection Flow
            axs00[idx, b].bar(
                np.linspace(1, max_time, max_time), self.rejection.iloc[:, target]
            )
            axs00[idx, b].title.set_text("Rejection in [{}]".format(b + 1))
            idx += 1

            # Inflow
            axs00[idx, b].bar(
                np.linspace(1, max_time, max_time), self.max_inflow.iloc[:, target]
            )
            axs00[idx, b].title.set_text("Inflow in [{}]".format(b + 1))

            for idx in range(0, 5):
                axs00[idx, b].set_xticks(np.linspace(1, max_time, max_time))
                for iter in range(max_time):
                    if axs00[idx, b].containers[0].datavalues[iter] != 0:
                        axs00[idx, b].text(
                            x=axs00[idx, b].containers[0].patches[iter].xy[0],
                            y=axs00[idx, b].containers[0].datavalues[iter],
                            s=f"{axs00[idx, b].containers[0].datavalues[iter]}",
                        )
        #
        # for a in range(1):
        #     # Efficiency
        #     idx = 0
        #     axs01[idx].scatter(
        #         np.linspace(1, len(self.eff_store), len(self.eff_store)),
        #         self.eff_store,
        #         marker="^",
        #         c="g",
        #     )
        #     axs01[idx].plot(
        #         np.linspace(1, len(self.eff_store), len(self.eff_store)),
        #         self.eff_store,
        #         "g-.",
        #     )
        #     if x_stick_on:
        #         xaxis = [
        #             "[{}, {}]".format(i, mmax - i)
        #             for i in range(mmax - upper_bound, upper_bound + 1)
        #         ]
        #         axs01[idx].set_xticks(
        #             np.linspace(1, len(self.eff_store), len(self.eff_store)), xaxis
        #         )
        #     axs01[idx].title.set_text("Utilization")
        #     idx += 1
        #
        #     # Preference
        #     axs01[idx].scatter(
        #         np.linspace(1, len(self.pref_store), len(self.pref_store)),
        #         self.pref_store,
        #         marker="^",
        #         c="r",
        #     )
        #     axs01[idx].plot(
        #         np.linspace(1, len(self.pref_store), len(self.pref_store)),
        #         self.pref_store,
        #         "r-.",
        #     )
        #     if x_stick_on:
        #         axs01[idx].set_xticks(
        #             np.linspace(1, len(self.pref_store), len(self.pref_store)), xaxis
        #         )
        #     axs01[idx].title.set_text("Satisfaction")
        #     idx += 1
        #
        #     # Target
        #     axs01[idx].scatter(
        #         np.linspace(1, len(self.target_store), len(self.target_store)),
        #         self.target_store,
        #         marker="^",
        #         c="purple",
        #     )
        #     axs01[idx].plot(
        #         np.linspace(1, len(self.target_store), len(self.target_store)),
        #         self.target_store,
        #         color="purple",
        #         linestyle="-.",
        #     )
        #     if x_stick_on:
        #         axs01[idx].set_xticks(
        #             np.linspace(1, len(self.target_store), len(self.target_store)),
        #             xaxis,
        #         )
        #     axs01[idx].title.set_text("Objective")
        #     idx += 1
        #
        #     if x_stick_on:
        #         # Choice Changing
        #         lot_ratio = [
        #             i / mmax for i in range(mmax - upper_bound, upper_bound + 1)
        #         ]
        #         axs01[idx].scatter(
        #             np.linspace(1, len(lot_ratio), len(lot_ratio)),
        #             lot_ratio,
        #             marker="*",
        #             c="yellowgreen",
        #         )
        #         axs01[idx].plot(
        #             np.linspace(1, len(lot_ratio), len(lot_ratio)),
        #             lot_ratio,
        #             color="yellowgreen",
        #             linestyle="-.",
        #         )
        #         axs01[idx].scatter(
        #             np.linspace(1, len(lot_ratio), len(lot_ratio)),
        #             self.choice_num_store,
        #             marker="^",
        #             c="olive",
        #         )
        #         axs01[idx].plot(
        #             np.linspace(1, len(lot_ratio), len(lot_ratio)),
        #             self.choice_num_store,
        #             color="olive",
        #             linestyle="-.",
        #         )
        #         if x_stick_on:
        #             axs01[idx].set_xticks(
        #                 np.linspace(1, len(self.pref_store), len(self.pref_store)),
        #                 xaxis,
        #             )
        #         axs01[idx].title.set_text("Choice changing ratio (Choices on 0)")

        fig.suptitle("Allocation results")
        plt.show()

    # def figure_display(self, x_stick_on):
    #     def annotate_axes(ax, text, fontsize=18):
    #         ax.text(
    #             0.5,
    #             0.5,
    #             text,
    #             transform=ax.transAxes,
    #             ha="center",
    #             va="center",
    #             fontsize=fontsize,
    #             color="darkgrey",
    #         )
    #
    #     fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    #     gs0 = fig.add_gridspec(1, 1)
    #
    #     gs01 = gs0[0].subgridspec(2, 1)
    #     axs01 = gs01.subplots()
    #
    #     for a in range(1):
    #         # Efficiency
    #         idx = 0
    #         axs01[idx].scatter(
    #             np.linspace(1, len(self.eff_store), len(self.eff_store)),
    #             self.eff_store,
    #             marker="^",
    #             c="g",
    #         )
    #         axs01[idx].plot(
    #             np.linspace(1, len(self.eff_store), len(self.eff_store)),
    #             self.eff_store,
    #             "g-.",
    #         )
    #         if x_stick_on:
    #             xaxis = [
    #                 "[{}, {}]".format(i, mmax - i)
    #                 for i in range(mmax - upper_bound, upper_bound + 1)
    #             ]
    #             axs01[idx].set_xticks(
    #                 np.linspace(1, len(self.eff_store), len(self.eff_store)), xaxis
    #             )
    #         axs01[idx].title.set_text("Utilization")
    #         idx += 1
    #
    #         # Preference
    #         axs01[idx].scatter(
    #             np.linspace(1, len(self.pref_store), len(self.pref_store)),
    #             self.pref_store,
    #             marker="^",
    #             c="r",
    #         )
    #         axs01[idx].plot(
    #             np.linspace(1, len(self.pref_store), len(self.pref_store)),
    #             self.pref_store,
    #             "r-.",
    #         )
    #         if x_stick_on:
    #             axs01[idx].set_xticks(
    #                 np.linspace(1, len(self.pref_store), len(self.pref_store)), xaxis
    #             )
    #         axs01[idx].title.set_text("Satisfaction")
    #
    #     fig.suptitle("Allocation results")
    #     plt.show()

    def main(self, m, slots):
        self.arrangement(np.zeros(slots), 0, m)  # m is total chargers, n is slots
        print(
            "minimum possible rejection is: {}".format(
                self.expected_minimum_rejection()
            )
        )
        self.result_display()
        self.figure_display(x_stick_on)
        print()


# Global Variables
m = config.num_cars  # the number of cars
slots = config.total_parking_lot  # the number of parking lots
max_time = config.observation_time  # total time observed
mmax = config.total_chargers  # total number of charger
upper_bound = config.upper_bound_per_parking_lot  # max number in each parking lots
dist_base = config.dist_base
dist_change = config.dist_change
price_base = config.price_base
price_change = config.price_change
charging_time_switch = config.charging_time_switch
x_stick_on = config.x_stick_on
price_gameday_multiple = config.price_gameday_multiple
dist_gameday_multiple = config.dist_gameday_multiple
