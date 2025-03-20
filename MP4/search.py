import heapq

def best_first_search(starting_state):
    visited_states = {starting_state: (None, 0)}

    frontier = []
    heapq.heappush(frontier, starting_state)
    
    while(len(frontier) != 0):
        current_state = heapq.heappop(frontier)
        if(current_state.is_goal()):
            path = backtrack(visited_states, current_state)
            return path
        neighbors = current_state.get_neighbors()
        for neighbor in neighbors:
            # gotta keep track of the parent somehow
            if neighbor in visited_states:
                if (neighbor.dist_from_start < visited_states[neighbor][1]):
                    visited_states[neighbor] = (current_state, neighbor.dist_from_start)
                    heapq.heappush(frontier, neighbor) # Node Resurrection
            else:
                heapq.heappush(frontier, neighbor)
                visited_states[neighbor] = (current_state, neighbor.dist_from_start)
    
    # if you do not find the goal return an empty list
    return []

def backtrack(visited_states, goal_state):
    path = []
    # Your code here ---------------
    for state in visited_states:
        if state.is_goal():
            path.append(state)
            break
    goal_state = path[0]
    parent_state = visited_states[goal_state][0]
    while(visited_states[parent_state][1] != 0):
        path.append(parent_state)
        parent_state = visited_states[parent_state][0]
    # Deal with start_state
    path.append(parent_state)
    path.reverse()
    # ------------------------------
    return path