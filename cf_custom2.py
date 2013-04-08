import os
import math
import sys
from heapq import nlargest
from collections import defaultdict
if len(sys.argv) != 4:
    print "arguments: K data_directory query_file_name output_file_path"
K = int(sys.argv[1])
#DATA_PATH = "/Users/huanchen/Documents/cf_data/download_sample/training_set"
#QUERY_PATH = "/Users/huanchen/Documents/cf_data/download_sample/queries.txt"
#OUTPUT_PATH = "/Users/huanchen/Documents/cf_data/custom2_sim_full"+str(K)
DATA_PATH = sys.argv[2]
QUERY_PATH = sys.argv[3]
OUTPUT_PATH = sys.argv[4]
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
user_rating_ave = defaultdict(float)
user_rating_norm = defaultdict(float)
movie_rating_ave = defaultdict(float)
movie_rating_norm = defaultdict(float)
# do centering and normalization
for user in user_to_movie_rating:
    movies = user_to_movie_rating[user]
    cur_ave = float(sum(movies.values()))/len(movies)
    user_rating_ave[user] = cur_ave
    norm = 0.0
    for m in movies:
        movies[m] -= cur_ave
        norm += movies[m]**2
    norm = math.sqrt(norm)
    user_rating_norm[user] = norm
    for m in movies:
        if norm != 0:
            movies[m] /= norm
for movie in movie_to_user_rating:
    users = movie_to_user_rating[movie]
    cur_ave = float(sum(users.values()))/len(users)
    movie_rating_ave[movie] = cur_ave
    norm = 0.0
    for u in users:
        users[u] -= cur_ave
        norm += users[u]**2
    norm = math.sqrt(norm)
    movie_rating_norm[movie] = norm
    for u in users:
        if norm != 0:
            users[u] /= norm

# helper functions for later use
# compute the distance
def compute_dist(cur, neighbors):
    result = 0.0
    for m in cur.iterkeys():
        if m in neighbors:
            result += cur[m]* neighbors[m]
    return result

#get the k nearest neighbors
def get_k_nearest(cur, k, is_user):
    if is_user:
        work_set = user_to_movie_rating
        inverted_list = movie_to_user_rating
    else:
        work_set = movie_to_user_rating
        inverted_list = user_to_movie_rating
    cur_set = work_set[cur]
    s = set()
    for key1 in cur_set:
        for key2 in inverted_list[key1]:
            s.add(key2)
            # compute the distance
    def get_dist():
        for key in s:
            dist = compute_dist(cur_set, work_set[key])
            yield key, dist
    k_nearest = nlargest(k, get_dist(), key=lambda x:x[1])
    return k_nearest

# get the weighted mean
def get_rating(k_nn_movie, k_nn_user):
    result = 0.0
    total_weight = 0.0;
    # get the set of related movies or users
    related_set = set()
    for u_w in k_nn_user:
        u = u_w[0]
        for m_w in k_nn_movie:
            m = m_w[0]
            weight = m_w[1] * u_w[1]
            if u in movie_to_user_rating[m]:
                rating = movie_to_user_rating[m][u]
            else:
                rating = 0
            result += rating * weight
            total_weight += weight
    if total_weight != 0:
        result /= total_weight
    return result

# do prediction
k = K
f = open(OUTPUT_PATH, "w")
cache_movie = defaultdict(dict)
cache_user = defaultdict(dict)
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
        if movie_id in cache_movie:
            k_nearest_movie = cache_movie[movie_id]
        else:
            k_nearest_movie = get_k_nearest(movie_id, k, False)
            cache_movie[movie_id] = k_nearest_movie
    else:           # if this is a user id
        user_id = line[:-1]
        # for user-user
        if user_id in cache_user:
            k_nearest_user = cache_user[user_id]
        else:
            k_nearest_user = get_k_nearest(user_id, k, True)
            cache_user[user_id] = k_nearest_user
        rating = get_rating(k_nearest_movie,k_nearest_user)
        rating = rating * movie_rating_norm[movie_id] + movie_rating_ave[movie_id]
        f.write(str(rating)+"\n")
        #print "movie_id: " + movie_id + " user_id: " + user_id + " rating: " + str(rating)+"\n"
f.close()