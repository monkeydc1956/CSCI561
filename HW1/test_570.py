import  math
def countTicketCombinations(n):
    maxi = [0] * (n + 1)
    maxi[0] = 1
    for i in range(1, n + 1):
        if i >= 1:
            maxi[i] += maxi[i - 1]
        if i >= 2:
            maxi[i] += maxi[i - 2]
    return maxi[n]


def minPackagesTwo(weights, W):
    weights = sorted(weights)
    mini = [float('inf')] * (W + 1)
    mini[0] = 0
    for i in range(1, W + 1):
        for j in range(len(weights)):
            if weights[j] <= i:
                mini[i] = min(mini[i], mini[i - weights[j]] + 1)
    return mini[W]


def minPackagesOne(weights, W):
    n = len(weights)
    weights = sorted(weights)
    mini = [[float('inf')] * (W + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        mini[i][0] = 0
    # for j in range(W + 1):
    #     mini[1][j] = j
    print(mini)
    for j in range(1, W + 1):
        for i in range(1, n + 1):
            mini[i][j] = min(mini[i - 1][j], mini[i][j - weights[i - 1]] + 1)
    print(mini)
    return mini[n][W]
weights = [2,1,3]
W = 10
print(minPackagesOne(weights,W))

# import heapq
#
# def minimizeCost(rod_lengths):
#     min_heap = rod_lengths[:]
#     heapq.heapify(min_heap)
#     total_cost = 0
#     while len(min_heap) > 1:
#         rod1 = heapq.heappop(min_heap)
#         rod2 = heapq.heappop(min_heap)
#         combined_rod = rod1 + rod2
#         total_cost += combined_rod
#         heapq.heappush(min_heap, combined_rod)
#         print(min_heap )
#     return max(total_cost,min_heap[0])


# def min_flights(N, a, b):
#     flights = 0
#     while a != b:
#         if a > b:
#             a //= 2
#         else:
#             a *= 2
#         flights += 1
#     return flights
#
# # Example usage:
# N = 10
# a = 3
# b = 9
# result = min_flights(N, a, b)
# print("Minimum number of flights:", result)