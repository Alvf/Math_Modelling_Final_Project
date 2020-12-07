'''
Elevator Simulation

Alvin Shi

Methods and objects meant to simulate the actions and strategies of an elevator
handling queries from passengers
'''

import random
import itertools
from sympy.utilities.iterables import multiset_permutations
import numpy as np

class Elevator(object):

    def __init__(self, floor, speed = 1):
        '''
        an elevator keeps track of which floor it is on and how fast it is

        Parameters: 
            floor (int): the starting floor of the elevator
            speed (float): the speed of the elevator
        '''
        self.floor = floor
        self.speed = speed

    def time_move(self, target):
        '''
        Moves the elevator from its current floor to the target
        floor and returns the time elapsed
        
        Inputs:
            Target (int): the target floor
        
        Returns:
            The time elapsed to move the elevator from its current floor
            to the target
        '''
        old_floor = self.floor
        self.floor = target
        return abs(target - old_floor) / self.speed

class Passenger(object):

    def __init__(self, id, N, Q):
        '''
        a passenger keeps track of their queries throughout the day

        Parameters: 
            id (token): a token of identification for the passenger
            N (int): the amount of floors in the passenger's building
            Q (int): the amount of queries the passenger makes/day
        '''
        self.queries = []
        self.id = id
        self.make_queries(Q, N)
        self.on_request = 0

    def __query(self, N):
        '''
        Adds a single new query to the queries list of the passenger

        Inputs:
            N (int): the maximum height of the building
        '''
        if self.queries == []:
            start_floor = random.randint(1, N)
            end_floor = random.randint(1, N)
            while end_floor == start_floor:
                end_floor = random.randint(1, N)
            self.queries.append((start_floor, end_floor))
        else:
            _, start_floor = self.queries[-1]
            end_floor = random.randint(1, N)
            while end_floor == start_floor:
                end_floor = random.randint(1, N)
            self.queries.append((start_floor, end_floor))
    
    def __recluse_query(self, N, house):
        '''
        Adds one home-and-back query to the queries list
        also clears old query list

        Inputs:
            N (int): the maximum height of the building
            house (int): the story that the passenger's residence is on
        '''  
        if self.queries == []:
            self.queries.append((house, 1))
        else:
            _, start_floor = self.queries[-1]
            if start_floor == house:
                self.queries.append((house, 1))
            else:
                self.queries.append((1, house))
        
    def make_recluse_queries(self, num_queries, N, house):
        '''
        Adds a number of home-and-back queries to the queries list
        also clears old query list

        Inputs:
            num_queries (int): the number of queries to add
            N (int): the maximum height of the building
            house (int): the story that the passenger's residence is on
        '''
        self.queries = []
        for _ in range(num_queries):
            self.__recluse_query(N, house)

    def make_queries(self, num_queries, N):
        '''
        Adds a number of queries to the queries list of the passenger

        Inputs:
            num_queries (int): the number of queries to add
            N (int): the maximum height of the building
        '''
        for _ in range(num_queries):
            self.__query(N)
    
    def num_queries(self):
        '''
        tells how m any queries there are in the queries list

        Returns: the number of queries in the query list
        '''
        return len(self.queries)

    def next_query(self):
        '''
        returns the next query in the query list and advances the marker

        Returns: the next query in the query list
        '''
        if self.on_request >= self.num_queries():
            print("Passenger {} is already done!".format(self.id))
            return
        query = self.queries[self.on_request]
        self.on_request += 1
        return query
    
    def __repr__(self):
        '''
        returns the id, the number of queries, and which query they're on
        '''
        return "{} has {} queries and is on query index {}".format(self.id, self.num_queries(), self.on_request)


def passenger_query_rep(passengers):
    '''
    Takes a list of passengers and returns a list of passengers with
    repetition depending on how long their query lists are

    Inputs:
        N (int): the number of stories the building has
        passengers (list of Passengers): the passengers

    Returns:
        id: total time spent in transit for the day
    '''
    rep_list = []
    for ps in passengers:
        for _ in range(ps.num_queries()):
            rep_list.append(ps)
    return rep_list


def graph_sim_recluse_grow_random(N, points, house, trials):
    '''
    Preps two numpy arrays for graphing; A recluse passenger 
    dominates over a random passenger

    Inputs:
        N(int): building height
        points(int): total # data points at end
        house(int): story of recluse's house
        trials(int): number of trials to sim
    
    Returns:
        dictionary passenger id: average amount of time spent in transit/query
    '''
    P_random = []
    P_recluse = []
    i = 1
    while len(P_random) < points:
        final_totals = {}
        for _ in range(trials):
            p_rand = Passenger('r', N, 1)
            p_recluse = Passenger('e', N, i)
            p_recluse.make_recluse_queries(i, N, house)
            day_totals = simulate_day(N, [p_rand, p_recluse])
            for key, val in day_totals.items():
                final_totals[key] = final_totals.get(key, 0) + val
        final_totals['r'] /= trials
        final_totals['e'] /= trials
        final_totals['e'] /= i
        P_random.append(final_totals['r'])
        P_recluse.append(final_totals['e'])
        i += 1
    return(P_random, P_recluse)


def graph_sim_random_grow_recluse(N, points, house, trials):
    '''
    Preps two numpy arrays for graphing; A random passenger 
    dominates over a recluse passenger

    Inputs:
        N(int): building height
        points(int): total # data points at end
        house(int): story of recluse's house
        trials(int): number of trials to sim
    
    Returns:
        dictionary passenger id: average amount of time spent in transit/query
    '''
    P_random = []
    P_recluse = []
    i = 1
    while len(P_random) < points:
        final_totals = {}
        for _ in range(trials):
            p_rand = Passenger('r', N, i)
            p_recluse = Passenger('e', N, 1)
            p_recluse.make_recluse_queries(1, N, house)
            day_totals = simulate_day(N, [p_rand, p_recluse])
            for key, val in day_totals.items():
                final_totals[key] = final_totals.get(key, 0) + val
        final_totals['r'] /= trials
        final_totals['e'] /= trials
        final_totals['r'] /= i
        P_random.append(final_totals['r'])
        P_recluse.append(final_totals['e'])
        i += 1
    return(P_random, P_recluse)


def simulate_day(N, passengers):
    '''
    Takes an array of passengers and simulates one day where each passenger
    entirely completes the queries in their list

    Inputs:
        N (int): the number of stories the building has
        passengers (list of Passengers): the passengers

    Returns:
        dictionary {id: total time spent in transit for the day}
    '''
    return_dict = {}

    rep_list = passenger_query_rep(passengers)
    ele = Elevator(1)

    random.shuffle(rep_list)
    for ps in rep_list:
        start, end = ps.next_query()
        return_dict[ps.id] = return_dict.get(ps.id, 0) + ele.time_move(start) + ele.time_move(end)
    return return_dict


def dumb_prediction_analysis(N, query_list, trials):
    '''
    outputs both the prediction and the simulation for dumb elevator results
    in terms average total time spent in transit/day
    '''
    query_amounts = token_scrub(query_list)
    predictions = expected_dumb_day(N, query_list)
    results = simulate_dumb_day(N, query_list, trials)
    for id in predictions.keys():
        print("Passenger {}: expected {} | got {}".format(id, predictions[id], results[id]))
    print("Expected average transit/query | Experimental avg wait/query")
    # really just starting to spaghetti for this display method but whatever
    for id in predictions.keys():
        print("Passenger {}: expected {} | got {}".format(id, predictions[id] / query_amounts[id], results[id] / query_amounts[id]))
    
def dumb_wait_day(N, query_list):
    '''
    With passengers as described in the query list, simulate elevator
    behavior in a day and then report back the average total amount of time 
    spent in transit for each passsenger as a dictionary
    
    Inputs:
        N (int): the number of stories in the building
        query_list: the list that describes the queries in a day
    
    Returns: a dictionary with key: value pairs representing passenger: time 
        in transit the whole day
    '''
    travels = {}
    ele = Elevator(1)

    passengers = queries_to_passengers(N, query_list)
    passenger_appearance = []
    for passenger in passengers.values():
        for _ in range(passenger.num_queries()):
            passenger_appearance.append(passenger)
    random.shuffle(passenger_appearance)
    for passenger in passenger_appearance:
        id = passenger.id
        s, e = passenger.next_query()
        travels[id] = travels.get(id, 0) + ele.time_move(s) + ele.time_move(e)
    return travels


def simulate_dumb_day(N, query_list, trials):
    '''
    With passengers as described in the query list, simulate elevator
    behavior in a day for a certain number of trials and then report back
    the average total amount of time spent in transit for each passsenger
    as a dictionary
    
    Inputs:
        N (int): the number of stories in the building
        query_list: the list that describes the queries in a day
        trials: the number of trials to try before returning a time waited
    
    Returns: a dictionary with key: value pairs representing passenger: time 
        in transit the whole day
    '''
    running_totals = {}   
    for _ in range(trials):
        scratch_total = dumb_wait_day(N, query_list)
        for key, value in scratch_total.items():
            running_totals[key] = running_totals.get(key, 0) + value
    for key, value in running_totals.items():
        running_totals[key] = value / trials
    return running_totals


def token_scrub(query_list):
    '''
    From a list of tokens, generate a dictionary of token: #occurences pairs

    Inputs:
        query_list: the list of tokens that describes the queries in a day
    
    Returns: a dictionary of token: #occurences pairs
    '''
    return_dict = {}
    for id in query_list:
        return_dict[id] = return_dict.get(id, 0) + 1
    return return_dict


def queries_to_passengers(N, query_list):
    '''
    from a raw list of tokens, generate a corresponding passenger dictionary

    Inputs:
        N (int): the number of stories in the building
        query_list: the list that describes the queries in a day
    
    Returns: a dictionary of passengers with id: Passenger key/value pairs
    '''
    occurence_dict = token_scrub(query_list)
    return_dict = {}
    for id, queries in occurence_dict.items():
        return_dict[id] = Passenger(id, N, queries)
    return return_dict


def gen_graphing_list(N, Q1_upper):
    '''
    Generates a list of P1 times and P2 times illustrating average wait/query
    changes in an N-story building as P1 dominates the query list
    Inputs:
        Q1_upper: The upper bound for Q1 to take (1 <= Q1 <= Q1_upper)
        N: the amount of stories in the building
    
    Returns:
        Two numpy arrays: (P1 transit/query, P2 transit/query) as they vary.
    '''
    P1 = []
    P2 = []
    query_list = [1,2]
    for _ in range(Q1_upper):
        waits = expected_dumb_day_averages(N, query_list)
        P1.append(waits[1])
        P2.append(waits[2])
        query_list.append(1)
    return (np.array(P1), np.array(P2))


def gen_sym_graphing_list_sim(N, Q_upper,trials):
    '''
    Generates a list of P1 times and P2 times illustrating average wait/query
    changes in an N-story building as P1 and P2 queries grow equally fast
    Inputs:
        Q_upper: The upper bound for Q to take (1 <= Q <= Q_upper)
        N: the amount of stories in the building
        trials: the amount of trials before returning a final value for that
          query amount
    
    Returns:
        Two numpy arrays: (P1 transit/query, P2 transit/query) as they vary.
    '''
    P1 = []
    P2 = []
    query_list = [1,2]
    for n in range(Q_upper):
        waits = simulate_dumb_day(N, query_list,trials)
        P1.append(waits[1] / (n + 1))
        P2.append(waits[2] / (n + 1))
        query_list += [1,2]
    return (np.array(P1), np.array(P2))


def gen_sym_graphing_list(N, Q_upper):
    '''
    Generates a list of P1 times and P2 times illustrating average wait/query
    changes in an N-story building as P1 and P2 queries grow equally fast
    Inputs:
        Q_upper: The upper bound for Q to take (1 <= Q <= Q_upper)
        N: the amount of stories in the building
    
    Returns:
        Two numpy arrays: (P1 transit/query, P2 transit/query) as they vary.
    '''
    P1 = []
    P2 = []
    query_list = [1,2]
    for _ in range(Q_upper):
        waits = expected_dumb_day_averages(N, query_list)
        P1.append(waits[1])
        P2.append(waits[2])
        query_list += [1,2]
    return (np.array(P1), np.array(P2))

def expected_dumb_day_averages(N, query_list, speed = 1):
    '''
    Calculates the expected average time spent in transit/query
    for each individual passenger in the query list for a building of 
    hieght N with a dumb elevator

    Inputs:
        N (int): the building height
        query_list: list of passenger ids with repitition 
        speed: elevator speed
    
    Returns: dictionary with id: epxected avg wait/query pairs
    '''
    wait_times = {}
    ids = token_scrub(query_list)
    for id, num in ids.items():
        wait_times[id] = expected_total_waits(N, id, query_list, speed) / num
    return wait_times


def expected_dumb_day(N, query_list, speed = 1):
    '''
    Calculates the expected average total time spent in transit/day
    for each individual passenger in the query list for a building of 
    hieght N with a dumb elevator

    Inputs:
        N (int): the building height
        query_list: list of passenger ids with repitition 
        speed: elevator speed
    
    Returns: dictionary with id: epxected avg wait/day pairs
    '''
    wait_times = {}
    ids = list(token_scrub(query_list).keys())
    for id in ids:
        wait_times[id] = expected_total_waits(N, id, query_list, speed)
    return wait_times

def expected_average_wait(N, passenger, query_list, speed = 1):
    '''
    Calculates the expected average transit time of a passenger
    in query_list for the day in a building of height N with a dumb elevator

    Inputs:
        N (int): the building height
        passenger: the id that represents the passenger in question
        query_list (list of strings): List of passenger names with repitition
            representing the amount of queries made in a day
        speed: speed of the elevator
    
    Returns: the expected average amount of time spent in transit for the first 
        passenger in the query list
    '''
    Q_i = query_list.count(passenger)
    return expected_total_waits(N, passenger, query_list, speed) / Q_i


def list_to_string(query_list):
    '''
    converts a list of elements into a big concatenated string
    for use in multiset_permutations

    Inputs:
        query_list (list): a list of stuff to turn into a big string
    '''
    ret_string = ''
    for x in query_list:
        ret_string += str(x)
    return ret_string


def expected_total_waits(N, passenger, query_list, speed = 1):
    '''
    Calculates the expected total transit time of the id
    in query_list for the day in a building of height N with a dumb elevator

    Inputs:
        N (int): the building height
        passenger: the id that represents the passenger in question
        query_list: the list that describes the queries in a day
        speed: speed of the elevator
    
    Returns: the expected total amount of time spent in transit for the first 
        passenger in the query list
    '''
    assert passenger in query_list
    pa = str(passenger)
    query_string = list_to_string(query_list)
    query_perms = [''.join(i) for i in multiset_permutations(query_string)]
    perms_total = len(query_perms)
    total_firsts = 0
    total_crossovers = 0
    for perm in query_perms:
        total_firsts += in_first(pa, perm)
        total_crossovers += crossovers(pa, perm)
    first_wait_avg = total_firsts * (N - 1) / 2 / perms_total / speed
    cross_avg = total_crossovers * (N ** 2 - 1) / (3 * N) / perms_total / speed
    elevator_transit_avg = query_list.count(passenger) * (N + 1) / 3 / speed
    return first_wait_avg + cross_avg + elevator_transit_avg

def crossovers(passenger, perm):
    '''
    Calculates number of crossover situations in the permutation

    Inputs:
        passenver (string): name of passenger of interest
        perm (string tuple): the permutation in question
    
    Returns: number of crossovers (as described in notes)
    '''
    total_cross = 0
    for i, p in enumerate(perm):
        if i == 0:
            continue
        if passenger == p != perm[i-1]:
            total_cross +=1
    return total_cross
            

def in_first(passenger, perm):
    '''
    Calculates if the passenger is first in the permutation and returns one
    if true. Returns 0 otherwise

    Inputs:
        passenger (string): name of passenger of interest
        perm (string tuple): the permutation in question

    Returns: 1 if passenger is first, 0 otherwise
    '''
    if passenger == perm[0]:
        return 1
    return 0

# Checkers to help convince myself that average distance between two
# uniformly randomly picked numbers b/w 1-N is (N ** 2 - 1) / (3 * N)

def distance_helper(N, trials):
    '''
    Pick two numbers uniformly randomly from 1, 2, ..., N 
    and return the distance average distance between them after 
    some amount of trials

    Inputs:
        N (int): the number of stories in the building
        trials: the number of trials
    
    Returns: the average distance apart
    '''
    running_total = 0
    for _ in range(trials):
        running_total += abs(random.randint(1, N) - random.randint(1, N))
    return running_total / trials

def check_distance_pred(max_N, trials):
    '''
    Checks my math for expected distance between two uniformly randomly picked
    1-N numbers

    Inputs:
        max_N (int): the maximum height to work up towards
        trials (int): the number of trials per building

    Returns: the average ratio after working through all N's and predictions
    '''
    running_ratio = 0
    for N in range(2, max_N + 1):
        running_ratio += distance_helper(N, trials) / ((N ** 2 - 1) / (3 * N))
    return running_ratio / (max_N - 1)

# Checkers to help convince myself that the average query distance is 
# (N + 1) / 3

def avg_query_dist(N, trials):
    '''
    numerically calculates an average query distance from some
    amount of trials

    Inputs:
        N (int): height of the building
        trials (int): number of trials

    Returns: average query distance ratio with prediction formula
    '''
    passenger = Passenger(1, N, trials)
    running_total = 0
    for start, end in passenger.queries:
        running_total += abs(start - end)
    return running_total / trials / ((N + 1) / 3)