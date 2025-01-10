from copy import deepcopy
from ast import literal_eval
import networkx as nx
import pygtrie as pt

### INPUTS ###

# The invalid squares are spaces by default
LETTERS_LEGACY = [
    ['c', 'e', 'r', 'h', 't'],
    ['e', ' ', 'x', ' ', 'f'],
    ['n', 'c', 'i', 's', 'e'],
    ['t', ' ', 'e', ' ', 'i'],
    ['a', 'u', 's', 'g', 'r']
]

# 0 --> grey, 1 --> yellow, 2 --> green
# The invalid squares are -1 by default
STATES_LEGACY = [
    [2,  1,  0,  1,  2],
    [0, -1,  2, -1,  0],
    [1,  0,  2,  0,  1],
    [0, -1,  1, -1,  0],
    [2,  0,  1,  0,  2]
]

LETTERS_DELUXE = [
    ['e', 'n', 'e', 'e', 'a', 'e', 'r'],
    ['d', ' ', 'l', ' ', 'n', ' ', 'g'],
    ['e', 'p', 'c', 's', 'p', 'p', 'd'],
    ['d', ' ', 'l', ' ', 'a', ' ', 'a'],
    ['s', 'd', 'a', 'n', 'r', 'e', 'w'],
    ['s', ' ', 'o', ' ', 'u', ' ', 'u'],
    ['r', 'v', 'e', 'e', 'l', 'o', 'e']
]

STATES_DELUXE = [
    [1,  1,  2,  0,  2,  0,  1],
    [1, -1,  0, -1,  0, -1,  0],
    [2,  0,  2,  1,  2,  0,  2],
    [0, -1,  2, -1,  2, -1,  0],
    [2,  0,  2,  0,  2,  0,  2],
    [1, -1,  0, -1,  0, -1,  0],
    [0,  0,  2,  1,  2,  0,  1]
]

# 0 --> Legacy, 1 --> Deluxe
PLAY_MODE = 0

#################################################################################################################################################

# Number of rows/cols
N_LEGACY = 5

N_DELUXE = 7

# Database of valid 5-letter words
with open("words.txt", "r") as file:
    WORDS_LEGACY = pt.CharTrie()
    for word in file:
        word = word.strip()
        WORDS_LEGACY[word] = True

with open("words_deluxe.txt", "r") as file:
    WORDS_DELUXE = pt.CharTrie()
    for word in file:
        word = word.strip()
        WORDS_DELUXE[word] = True

# Solve order: horizontals from top to bottom, then verticals from left to right
DIRS_LEGACY = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
    (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
    (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)
]

DIRS_DELUXE = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
    (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
    (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
    (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)
]

if PLAY_MODE == 0:
    LETTERS = LETTERS_LEGACY
    STATES = STATES_LEGACY
    N = N_LEGACY
    WORDS = WORDS_LEGACY
    DIRS = DIRS_LEGACY
    # Start with 15 swaps - optimal solution is in 10 swaps (5 remaining)
    GOAL = 10
else:
    LETTERS = LETTERS_DELUXE
    STATES = STATES_DELUXE
    N = N_DELUXE
    WORDS = WORDS_DELUXE
    DIRS = DIRS_DELUXE
    # Start with 25 swaps - optimal solution is in 20 swaps (5 remaining)
    GOAL = 20

def populate_cands(cands):
    used = set((i,j) for j in range(N) for i in range(N) if i%2 != 1 or j%2 != 1)
    outs = deepcopy(cands)

    # Populate greens
    for i in range(N):
        for j in range(N):
            if STATES[i][j] == 2:
                cands[i][j].add((i,j))
                used.remove((i,j))

    # Populate yellows and greys
    for i,j in list(used):
        if STATES[i][j] == -1: continue

        cy = set()
        cg = set(used)

        mi, mj = i % 2, j % 2
        if STATES[i][j] == 1 and mi + mj == 1:
            # Check for the funky dingo situation
            dingo = False

            if mj == 1:
                # If on a row, iterate through all cols and check for a match
                for k in range(0, N, 2):
                    for l in range(N):
                        if l == i: continue
                        if STATES[l][k] == 1 and LETTERS[l][k] == LETTERS[i][j]:
                            dingo = True
                            break
                    if dingo: break
            else:
                # If on a col, iterate through all rows and check for a match
                for k in range(0, N, 2):
                    for l in range(N):
                        if l == j: continue
                        if STATES[k][l] == 1 and LETTERS[k][l] == LETTERS[i][j]:
                            dingo = True
                            break
                    if dingo: break

            if dingo:
                # This letter can go anywhere (except its own position ofc)!
                cd = set(used)
                cd.remove((i,j))
                outs[i][j] = cd
                continue

        if mi == 0:
            for k in range(N):
                if STATES[i][k] != 2 and k != j: cy.add((i,k))
                cg.discard((i,k))
        if mj == 0:
            for k in range(N):
                if STATES[k][j] != 2 and k != i: cy.add((k,j))
                cg.discard((k,j))

        outs[i][j] = cg if STATES[i][j] == 0 else cy

    for i in range(N):
        for j in range(N):
            for k,l in list(outs[i][j]):
                if LETTERS[i][j] != LETTERS[k][l]:
                    cands[k][l].add((i,j))

def enumerate_max_matchings(G):
    def is_valid(curr_matching, e):
        u,v = e
        for x,y in curr_matching.items():
            if u == x or u == y or v == x or v == y:
                return False
        return True
    
    def backtrack(curr_matching, edges, idx, max_size):
        if len(curr_matching) == max_size:
            yield curr_matching
            return
        
        for i in range(idx, len(edges)):
            e = edges[i]
            if is_valid(curr_matching, e):
                u,v = e
                curr_matching[u] = v
                yield from backtrack(curr_matching, edges, i + 1, max_size)
                curr_matching.pop(u)

    edges = sorted(G.edges())
    max_size = len(nx.algorithms.bipartite.maximum_matching(G, top_nodes=[u for u in G.nodes() if "_out" in u])) // 2
    return backtrack({}, edges, 0, max_size)

def max_matchings_approx(G, k=100000):
    top_nodes=[u for u in G.nodes() if "_out" in u]
    for _ in range(k):
        matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=top_nodes)
        yield matching
    return

def maximum_cycle_cover(G):
    # Create bipartite graph for matching
    B = nx.DiGraph()
    for u, v in G.edges():
        B.add_edge(f"{u}_out", f"{v}_in")
    
    # Find/approximate maximum matching
    res, len_res = [], 0

    matching_method = enumerate_max_matchings if PLAY_MODE == 0 else max_matchings_approx
    for matching in matching_method(B):
        # Convert matching into cycles
        visited = set()
        cycles = []

        for node in G.nodes():
            if node not in visited:
                cycle = []
                current = f"{node}_out"

                while current not in visited:
                    visited.add(current)
                    # Extract original node name
                    cycle.append(current.split('_')[0])
                    # Convert to corresponding out node
                    next_node = f"{matching.get(current).split('_')[0]}_out"

                    if next_node:
                        current = next_node
                    else:
                        break

                # Close the cycle
                if cycle:
                    cycles.append(cycle)

        if len(cycles) > len_res:
            res = cycles
            len_res = len(cycles)

        # If the number of swaps is the goal, we've found our solution!
        swaps = sum([len(r)-1 for r in res])
        if swaps == GOAL:
            break
    
    return res

def solve(cands, res, word, visited, idx):
    if idx == len(DIRS):
        # Print S
        for r in res: print(r)
        print()

        # STEP 2: Given solution S, construct a graph G, where any node N in G (corresponding to a position in the waffle) points to all other
        # positions P that satisfy S (i.e. positions whose states turn green when the letters in N and P are swapped)
        G = nx.DiGraph()
        e = []

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        if LETTERS[i][j] != ' ' and STATES[i][j] != 2 and STATES[k][l] != 2 and LETTERS[i][j] == res[k][l]:
                            e.append(((i,j), (k,l)))

        G.add_edges_from(e)

        # STEP 3: Find the Maximum Cycle Cover (MCC) of G (i.e. break G into the maximum number of cycles that exactly span G; every cycle C in
        # this cover is a set of swaps that constructs a partial solution of S - maximizing the cycles minimizes the swaps, as the swap that
        # completes a cycle is the most efficient move in Waffle: a move where both letters turn green)
        cycles = maximum_cycle_cover(G)

        # STEP 4: Traverse the MCC, tracking each swap
        i = 1
        for c in cycles:
            for j in range(1, len(c)):
                k,l = literal_eval(c[j-1])
                x,y = literal_eval(c[j])
                print(f"Move {i}: Swap {c[0]} [{LETTERS[k][l]}] and {c[j]} [{LETTERS[x][y]}]")
                i += 1
            print()

        # Print the swaps

        return True
    
    # STEP 1: Given the letters and their corresponding states, identify a possible solution S (i.e. a configuration that satisfies the waffle)
    i,j = DIRS[idx]
    m = idx % N

    # Verify a word as characters are being added (via Trie)
    if idx > 0:
        # Check if full word is a key in the Trie
        if m == 0:
            if not WORDS.has_key(word):
                return False
            word = ""
        # Otherwise check if prefix is a subtrie
        else:
            if not WORDS.has_subtrie(word):
                return False

    if visited[i][j]:
        t = res[i][j]
        word += t

        if solve(cands, res, word, visited, idx + 1):
            return True
        
        word = word[:-1]
    else:
        # Brute force backtracking alg
        for k,l in list(cands[i][j]):
            # Update
            t = res[i][j]
            res[i][j] = LETTERS[k][l]
            word += LETTERS[k][l]
            visited[i][j] = True

            updates = []
            for x in range(N):
                for y in range(N):
                    if (k,l) in cands[x][y] and (x,y) != (i,j):
                        cands[x][y].remove((k,l))
                        updates.append((x,y))

            # Recurse
            if solve(cands, res, word, visited, idx + 1):
                return True
            
            # Backtrack
            res[i][j] = t
            word = word[:-1]
            visited[i][j] = False

            for x,y in updates:
                cands[x][y].add((k,l))

    return False

if __name__ == "__main__":
    cands = [[set() for _ in range(N)] for _ in range(N)]
    populate_cands(cands)
    if not solve(cands, deepcopy(LETTERS), "", [[False for _ in range(N)] for _ in range(N)], 0):
        print("No solution found. Is your input correct?")
