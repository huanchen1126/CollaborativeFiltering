# algorithm without normalization
import os
import math
import sys
from heapq import nlargest
from collections import defaultdict
if len(sys.argv) != 5:
    print "arguments: K user_or_movie data_directory query_file_name output_file_path"
    sys.exit()
K = int(sys.argv[1])
user_or_movie = sys.argv[2]
if user_or_movie == "user":
    IS_USER = True
else:
    IS_USER = False

DATA_PATH = sys.argv[3]
QUERY_PATH = sys.argv[4]
OUTPUT_PATH = sys.argv[5]
# load data into memory
dir_name = DATA_PATH
dir_files = os.listdir(dir_name)

user_to_movie_rating = defaultdict(dict)
movie_to_user_rating = defaultdict(dict)

for filename in dir_files:
    f = open(os.path.join(dir_name,filename), "r")
    lines = f.readlines()
    movie_name = lines[0][:-2]
    for line in lines[1:]:
        tmp = line[:-1].split(",")
        user_id = tmp[0]
        rating = tmp[1]
        user_to_movie_rating[user_id][movie_name] = int(rating)
        movie_to_user_rating[movie_name][user_id] = int(rating)
    f.close()
# do vector standardization
# cache the average and norm for later prediction
user_rating_norm = defaultdict(float)
movie_rating_norm = defaultdict(float)
# do centering and normalization
for user in user_to_movie_rating:
    movies = user_to_movie_rating[user]
    norm = 0.0
    for m in movies:
        movies[m] -= 3
        norm += movies[m]**2
    norm = math.sqrt(norm)
    user_rating_norm[user] = norm

for movie in movie_to_user_rating:
    users = movie_to_user_rating[movie]
    norm = 0.0
    for u in users:
        users[u] -= 3
        norm += users[u]**2
    norm = math.sqrt(norm)
    movie_rating_norm[movie] = norm
# helper functions for later use
# compute the distance
def compute_dist(cur, neighbors):
    result = 0.0
    for m in cur.iterkeys():
        if m in neighbors:
            result += cur[m]* neighbors[m]
    return result

# get the k nearest neighbors
def get_k_nearest(cur, k, is_user):
    if is_user:
        work_set = user_to_movie_rating
        inverted_list = movie_to_user_rating
        norm_set = user_rating_norm
    else:
        work_set = movie_to_user_rating
        inverted_list = user_to_movie_rating
        norm_set = movie_rating_norm
    cur_set = work_set[cur]
    candidate = set()
    for key1 in cur_set:
        for key2 in inverted_list[key1]:
            candidate.add(key2)
        # compute the distance
    def get_dist():
        for key in candidate:
            dist = compute_dist(cur_set, work_set[key])
            if norm_set[cur] != 0 and norm_set[key] != 0:
                dist /= norm_set[cur]*norm_set[key]
            if key != cur:
                yield key, dist
    k_nearest = nlargest(k, get_dist(), key=lambda x:x[1])
    return k_nearest

# get the weighted mean
def get_rating(k_nearest, is_user):
    if is_user:
        work_set = user_to_movie_rating
    else:
        work_set = movie_to_user_rating
    result = defaultdict(float)
    total_weight = 0;
    # get the set of related movies or users
    related_set = set()
    for neighbor in k_nearest:
        key = neighbor[0]
        value = work_set[key]
        for id in value:
            related_set.add(id)
        # calculate the ratings
    for neighbor in k_nearest:
        key = neighbor[0]
        value = work_set[key]
        for id in value:
            if id in related_set:
                result[id] += neighbor[1]*value[id]
            else:
                result[id] += neighbor[1]*3
        total_weight += neighbor[1]
        # get the weighted average
    for id in result:
        if total_weight != 0:
            result[id] /= total_weight
    return result

# do prediction
k = K
f = open(OUTPUT_PATH, "w")
cache = defaultdict(dict)
# read queries
query_file_name = QUERY_PATH
query_f = open(query_file_name, "r")
movie_id = ""
user_id = ""
lines = query_f.readlines()
for line in lines:
    ind = line.find(":")
    if ind != -1:   # if this is a movie id
        movie_id = line[:ind]
        f.write(movie_id+":\n")
        # if this is using movie-movie similarity
        if not IS_USER:
            if movie_id in cache:
                k_nearest = cache[movie_id]
            else:
                k_nearest = get_k_nearest(movie_id, k, IS_USER)
                cache[movie_id] = k_nearest
            ratings = get_rating(k_nearest, IS_USER)
    else:           # if this is a user id
        user_id = line[:-1]
        if not IS_USER:
            rating = ratings[user_id]
            rating = rating + 3
            f.write(str(rating) + "\n")
            # print "movie_id: " + movie_id + " user_id: " + user_id + " rating: " + str(rating) + "\n"
        else:
            if user_id in cache:
                k_nearest = cache[user_id]
            else:
                k_nearest = get_k_nearest(user_id, k, IS_USER)
                cache[user_id] = k_nearest
            ratings = get_rating(k_nearest, IS_USER)
            rating = ratings[movie_id]
            rating = rating + 3
            f.write(str(rating)+"\n")
            # print "movie_id: " + movie_id + " user_id: " + user_id + " rating: " + str(rating)+"\n"
f.close()