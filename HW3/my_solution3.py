import copy
def read_in_observation(pathname):
    states = []
    actions = []
    f = open(pathname)
    lines = f.readlines()
    for i in range(2, len(lines)-1):
        if i < len(lines)-1:
            line = lines[i].split(" ")
            states.append(str(line[0][1:-1]))
            actions.append(str(line[1][1:-2]))
    states.append(str(lines[-1][1:-2]))
    return states, actions

def read_in_state_observation_normalization(pathname, states_set):
    f = open(pathname)
    lines = f.readlines()
    default_weight = int(lines[1].split(" ")[-1])
    states_observations = {}
    flag = 2
    observations_set = set()
    for sigle_line in lines[2:]:
        obs = sigle_line.split(" ")[1][1:-1]
        observations_set.add(obs)
    while(flag < len(lines)):
        tmp_state = lines[flag].split(" ")[0][1:-1]
        line = lines[flag].split(" ")
        if tmp_state in states_observations.keys():
            tmp_set = states_observations[tmp_state]
        else:
            tmp_set = {}
        if len(line) == 3:
            tmp_set[line[1][1:-1]] = int(line[2])
        else:
            tmp_set[line[1][1:-1]] = default_weight
        flag += 1
        states_observations[tmp_state] = tmp_set
    # missing state
    for states_tmp in states_set:
        if states_tmp not in states_observations.keys():
            tmp_set_missed = {}
            for obs in observations_set:
                tmp_set[obs] = default_weight
            states_observations[states_tmp] = tmp_set_missed
    # missing observations for each state
    for key in states_observations.keys():
        for obs in observations_set:
            if obs not in states_observations[key].keys():
                states_observations[key][obs] = default_weight
    for key in states_observations.keys():
        tmp_base = 0
        for sub_key in states_observations[key].keys():
            tmp_base+= states_observations[key][sub_key]
        # for sub_key in states_observations[key].keys():
        #     states_observations[key][sub_key] = round(states_observations[key][sub_key]/tmp_base,2)
        states_observations[key]["Base"] = tmp_base
    return states_observations

def read_in_state_action_normalization(pathname,state_set):
    f = open(pathname)
    lines = f.readlines()
    default_weight = int(lines[1].split(" ")[-1])
    states_action = {}
    flag = 2
    # fix metrix
    actions_set = set()
    for sigle_line in lines[2:]:
        obs = sigle_line.split(" ")[1][1:-1]
        actions_set.add(obs)

    while(flag < len(lines)):
        tmp_state_action = lines[flag].split(" ")[0][1:-1]+lines[flag].split(" ")[1][1:-1]
        tmp_set = {}
        line = lines[flag].split(" ")
        if tmp_state_action in states_action.keys():
            tmp_set = states_action[tmp_state_action]
        if len(line) == 4:
            tmp_set[line[2][1:-1]] = int(line[3])
        else:
            tmp_set[line[2][1:-1]] = default_weight
        flag += 1
        states_action[tmp_state_action] = tmp_set
    # missing state_action
    for state_tmp in state_set:
        for actions_tm in actions_set:
            SA = state_tmp+actions_tm
            if SA not in states_action.keys():
                tmp_set = {}
                for sub_state_tmp in state_set:
                    if sub_state_tmp != state_tmp:
                        tmp_set[sub_state_tmp] = default_weight
                states_action[SA] = tmp_set
    # missing next state for SA
    for sa in states_action.keys():
        for state_tmp in state_set:
            if state_tmp not in states_action[sa].keys():
                tmp_set = states_action[sa]
                tmp_set[state_tmp] = default_weight
                states_action[sa] = tmp_set
    for key in states_action.keys():
        tmp_base = 0
        for sub_key in states_action[key].keys():
            tmp_base+= states_action[key][sub_key]
        # for sub_key in states_action[key].keys():
        #     states_action[key][sub_key] = round(states_action[key][sub_key]/tmp_base,2)
        states_action[key]["Base"] = tmp_base
    return states_action

def read_in_state_weight(pathname):
    f = open(pathname)
    lines = f.readlines()
    default_weight = int(lines[1].split(" ")[-1])
    states_wieghts = {}
    flag = 2
    tmp_base = 0
    while (flag < len(lines)):
        line = lines[flag].split(" ")
        if len(line) == 2:
            states_wieghts[line[0][1:-1]] = int(line[1])
            tmp_base += int(line[1])
        else:
            states_wieghts[line[0][1:-1]] = default_weight
            tmp_base += default_weight
        flag += 1
    states_set = set(states_wieghts.keys())
    states_wieghts["Base"] = tmp_base
    return states_wieghts, states_set

def forward_algorithm(length_actions, observations, actions, state_observations, states_action, states_wieghts):
    # states_length = len(states_wieghts)
    base = "Base"
    tmp_path = ""
    path = []
    aplha = {}
    states_names = set(states_wieghts.keys())
    states_names.remove("Base")
    states_actions_names = set(states_action.keys())
    for key in states_names:
        tmp_path = key
        observations_names = set(state_observations[key].keys())
        break
    observations_names.remove("Base")
    for i in range(length_actions+1):
        tmp_aplha = {}
        if i == 0:
            for state in states_names:
                tmp_aplha[state] = (state_observations[state][observations[0]]/state_observations[state][base]
                                    * states_wieghts[state]/states_wieghts[base])
        else:

            for state in states_names:
                tmp_value = 0
                for sub_state in states_names:
                    tmp_value += (aplha[i-1][sub_state]) *(states_action[sub_state+actions[i-1]][state]/states_action[sub_state+actions[i-1]][base]) *(state_observations[state][observations[i]]/state_observations[state][base]) * (10^i+1)
                tmp_aplha[state] = tmp_value
        aplha[i] = tmp_aplha
        print(aplha)
    for key in aplha.keys():
        tmp_value = 0
        for sub_key in aplha[key].keys():
            if aplha[key][sub_key] > tmp_value:
                tmp_value = aplha[key][sub_key]
                tmp_path = sub_key
        path.append(tmp_path)
    return path

def viterbi_algorithm(length_actions, observations, actions, state_observations, states_action, states_wieghts):
    base = "Base"
    path = {}
    aplha = {}
    states_names = set(states_wieghts.keys())
    states_names.remove("Base")
    for key in states_names:
        observations_names = set(state_observations[key].keys())
        break
    observations_names.remove("Base")
    for i in range(length_actions+1):
        tmp_aplha = {}
        if i == 0:
            tmp_path = {}
            for state in states_names:
                tmp_value = (state_observations[state][observations[0]]/state_observations[state][base] * states_wieghts[state]/states_wieghts[base])
                # tmp_value = (state_observations[state][observations[0]]*states_wieghts[state])
                tmp_aplha[state] = tmp_value
                tmp_path[state] = [state]
            path[i] = tmp_path
        else:
            tmp_path = {}
            for sub_state in states_names:
                flag_value = 0
                flag_state = ""
                for pre_flag in aplha[i-1].keys():# pre state
                    tmp_value = (aplha[i-1][pre_flag]) *(states_action[pre_flag+actions[i-1]][sub_state]/states_action[pre_flag+actions[i-1]][base]) *(state_observations[sub_state][observations[i]]/state_observations[sub_state][base]) #* (10**i)
                    # tmp_value = (aplha[i - 1][pre_flag]) * (states_action[pre_flag + actions[i - 1]][sub_state]) * (
                    #                         state_observations[sub_state][observations[i]])
                    if tmp_value > flag_value:
                        flag_value = tmp_value
                        flag_state = pre_flag
                z = copy.deepcopy(path[i-1][flag_state])
                z.append(sub_state)
                tmp_aplha[sub_state] = flag_value
                tmp_path[sub_state] = z
        path[i] = tmp_path
        aplha[i] = tmp_aplha
    res_value = 0
    res = []
    for key in aplha[length_actions].keys():
        if aplha[length_actions][key] > res_value:
            res_value = aplha[length_actions][key]
            res = path[length_actions][key]
    return res

def write_file(pathname,path:list):
    with open(pathname, "w") as f:
        f.write("states\n")
        f.write(str(len(path)))
        f.write("\n")
        for i in range(len(path)):
            f.write("\"")
            f.write(path[i])
            f.write("\"")
            if i < len(path)-1:
                f.write("\n")
    f.close()
    return
if __name__ == '__main__':
    observation_name = "observation_actions.txt"
    state_observation_weight_name = "state_observation_weights.txt"
    state_action_weight_name = "state_action_state_weights.txt"
    state_weight_name = "state_weights.txt"
    observations, actions = read_in_observation(observation_name)
    states_wieghts, states_set = read_in_state_weight(state_weight_name)
    state_observations = read_in_state_observation_normalization(state_observation_weight_name, states_set)
    # print(state_observations)
    states_action = read_in_state_action_normalization(state_action_weight_name, states_set)
    # print(states_action)
    # path = forward_algorithm(len(actions), observations, actions, state_observations, states_action, states_wieghts)
    path = viterbi_algorithm(len(actions), observations, actions, state_observations, states_action, states_wieghts)
    # write_file("states_text.txt",path)
    print(path)

