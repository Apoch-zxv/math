import scipy.io
import scipy.sparse
import numpy as np
import math
import collections
import matplotlib.pyplot as plt
import time
from numba import jit
from joblib import Parallel, delayed
import multiprocessing


def gaussian_kernel(vec1, vec2, sigma):
    return np.exp(-1 * (np.linalg.norm(vec1 - vec2, 2) ** 2) / (2 * (sigma ** 2)))


def build_kernel_matrix(data_mat, sigma):
    copy_mat = np.copy(data_mat)
    kernel_matrix = np.zeros((data_mat.shape[0], data_mat.shape[0]))
    for i, data in enumerate(data_mat):
        for j, second_data in enumerate(copy_mat):
            kernel_matrix[i][j] = gaussian_kernel(data, second_data, sigma)
    return kernel_matrix


def apply_solution(alphas, kernel_points, apply_on, sigma):
    res = 0
    for i, kernel_point in enumerate(kernel_points):
        res += gaussian_kernel(kernel_point, apply_on, sigma) * alphas[i]
    return res


def find_optimal_regression_naive(data_mat, solution_vec, sigma, lamda):
    print("Building kernel matrix")
    kernel_matrix = build_kernel_matrix(data_mat, sigma)
    print("Inversing")
    inverse = np.linalg.inv(kernel_matrix + np.identity(data_mat.shape[0]) * lamda)
    print("Computing dot")
    return inverse.dot(solution_vec)


def predict_solution_vec(alphas, kernel_points, sigma, data):
    res = np.array([0] * data.shape[0])
    for i, d in enumerate(data):
        res[i] = apply_solution(alphas, kernel_points, d, sigma)
    return res


def naive_way_simple_testing():
    learn_sizes = []
    diffs = []
    for learn_size in range(100, 300):
        data_mat = mat["X"][:learn_size]
        solution_mat = mat["Y"][:learn_size]
        sigma = mat["sigma"][0][0]
        optimal = find_optimal_regression_naive(data_mat, solution_mat, sigma, mat["lambda"])

        #validation_data = mat["X"][learn_size:]
        #validation_solution = mat["Y"][learn_size:]

        validation_data = mat["X"][learn_size: learn_size + 100]
        validation_solution = mat["Y"][learn_size: learn_size + 100]

        solution = predict_solution_vec(optimal, data_mat, sigma, validation_data)
        diff = np.linalg.norm(solution - np.transpose(validation_solution), 2) / np.linalg.norm(np.transpose(validation_solution), 2)
        learn_sizes.append(learn_size)
        diffs.append(diff)

    plt.plot(learn_sizes, diffs)
    plt.show()


def naive_way_simple_testing_single():
    learn_size = 100
    data_mat = mat["X"][:learn_size]
    solution_mat = mat["Y"][:learn_size]
    sigma = mat["sigma"][0][0]
    optimal = find_optimal_regression_naive(data_mat, solution_mat, sigma, mat["lambda"])

    #validation_data = mat["X"][learn_size:]
    #validation_solution = mat["Y"][learn_size:]

    validation_data = mat["X"][learn_size: learn_size + 100]
    validation_solution = mat["Y"][learn_size: learn_size + 100]

    solution = predict_solution_vec(optimal, data_mat, sigma, validation_data)
    diff = np.linalg.norm(solution - np.transpose(validation_solution), 2) / np.linalg.norm(np.transpose(validation_solution), 2)


def create_converted(data, random_ws, random_bs, s):
    sqrt_2 = np.sqrt(2)
    single_data_transform = [sqrt_2 * np.cos(data.dot(np.transpose(random_ws[i])) + random_bs[i]) for i in xrange(s)]
    return np.array(single_data_transform) * (1 / np.sqrt(s))


def build_z_matrix(data_mat, sigma, s):
    random_ws = [np.random.normal(0, np.array([1] * data_mat.shape[1]) * (sigma ** -2.)) for i in xrange(s)]
    random_bs = np.random.uniform(0, 2 * np.pi, s)

    z = [create_converted(data, random_ws, random_bs, s) for data in data_mat]

    # v = z[12]
    # print v
    # w = z[88]
    # print w
    # print v.dot(np.transpose(w))
    # print gaussian_kernel(v, w, sigma)
    return np.array(z), random_ws, random_bs


def create_fourrier_preconditioner_inner(data_mat, is_debug, is_woodbury, lamda, s, sigma,
                                         solution_mat, validation_data):
    if is_debug:
        print("Building Z matrix")
    Z, random_ws, random_bs = build_z_matrix(data_mat, sigma, s)
    for r in random_ws:
        print r
    if is_woodbury:
        if is_debug:
            print("Computing inverse with woodbury", Z.shape)
        z_transpose = np.transpose(Z)
        small_inv = np.linalg.inv(z_transpose.dot(Z) + lamda * np.identity(Z.shape[1]))
        optimal = (1. / lamda) * (solution_mat - Z.dot(small_inv.dot(z_transpose.dot(solution_mat))))
    else:
        if is_debug:
            print("Computing inverse without woodbury")
        preconditioner = Z.dot(np.transpose(Z)) + lamda * np.identity(Z.shape[0])
        inv = np.linalg.inv(preconditioner)
        optimal = inv.dot(solution_mat)
    if is_debug:
        print("Predicting solution")
    total = predict_solutions(Z, optimal, random_bs, random_ws, s, validation_data, sigma)
    solution = np.transpose(np.array(total))
    if is_debug:
        print(solution)
    return solution


# def predict_solutions(Z, optimal, random_bs, random_ws, s, validation_data, sigma):
#     total = [0] * len(validation_data)
#     for j, single_validation_data in enumerate(validation_data):
#         converted_validation_data = create_converted(single_validation_data, random_ws, random_bs, s)
#         res = sum([(optimal[i] * a.dot(np.transpose(converted_validation_data))) for i, a in enumerate(Z)])
#         total[j] = res
#         if j % 1000 == 0:
#             print 100 * float(j) / len(validation_data)
#     return total


def predict_solutions(Z, optimal, random_bs, random_ws, s, validation_data, sigma):
    total = [0] * len(validation_data)
    for j, single_validation_data in enumerate(validation_data):
        converted_validation_data = create_converted(single_validation_data, random_ws, random_bs, s)
        res = np.transpose(optimal).dot(Z.dot(np.transpose(converted_validation_data)))
        #res = sum([(optimal[i] * a.dot(np.transpose(converted_validation_data))) for i, a in enumerate(Z)])
        total[j] = res
    return total


# def predict_solutions_inner(Z, optimal, random_bs, random_ws, s, single_validation_data, sigma):
#     converted_validation_data = create_converted(single_validation_data, random_ws, random_bs, s)
#     #return sum([(optimal[i] * a.dot(np.transpose(converted_validation_data))) for i, a in enumerate(Z)])
#     return np.transpose(optimal).dot(Z.dot(np.transpose(converted_validation_data)))
#
#
# def predict_solutions(Z, optimal, random_bs, random_ws, s, validation_data, sigma):
#     args = [(Z, optimal, random_bs, random_ws, s, single_validation_data, sigma) for single_validation_data in validation_data]
#     total = v.map(predict_solutions_inner, args)
#     return np.array(list(total))


# def create_fourrier_preconditioner(limit, sigma, lamda, is_woodbury, s, is_debug=False):
#     limit = limit if limit != -1 else mat["X"].shape[0]
#     data_mat = mat["X"][:limit]
#     solution_mat = mat["Y"][:limit]
#
#     validation_data = mat["Xt"][:limit]  # [limit: limit + 100]
#     validation_solution = mat["Yt"][:limit] # [limit: limit + 100]
#
#     return create_fourrier_preconditioner_inner(data_mat, is_debug, is_woodbury, lamda, s, sigma, solution_mat, validation_data, validation_solution)


def create_converted_nystrom(data, r_inv, choosen_data, sigma):
    total = []
    for single_data in choosen_data:
        total.append(gaussian_kernel(data, single_data, sigma))
    total = np.array(total)
    return total.dot(r_inv)


def create_nystrom_preconfitioner_inner(data_mat, is_debug, is_woodbury, lamda, s, sigma, solution_mat, validation_data):
    for i in range(5):
        try:
            indexes = np.random.choice(range(data_mat.shape[0]), s, replace=False)
            # solution_mat = solution_mat[indexes]
            middle_mat = []
            more_index = indexes.copy()
            if is_debug:
                print("Building middle")
            for index in indexes:
                res = []
                for index2 in more_index:
                    print data_mat[index]
                    print data_mat[index2]
                    print gaussian_kernel(data_mat[index2], data_mat[index], sigma)
                    res.append(gaussian_kernel(data_mat[index2], data_mat[index], sigma))
                middle_mat.append(np.array(res))
            middle_mat = np.array(middle_mat)
            if is_debug:
                print("Doing cholesky")
            print middle_mat
            R = np.linalg.cholesky(middle_mat)
            break
        except Exception, e:
            if i == 4:
                raise
            else:
                print "Retrying"
                continue
    # we take the transpose because in the lectures it is A = L.T * L and in numpy A = L * L.T
    r_inv = np.linalg.inv(R.T)
    if is_debug:
        print("Building side", r_inv.shape)
    left_mat = []
    for index in indexes:
        row = data_mat[index]
        res = []
        for another_row in data_mat:
            res.append(gaussian_kernel(another_row, row, sigma))
        left_mat.append(np.array(res))
    left_mat = np.transpose(np.array(left_mat))
    Z = left_mat.dot(r_inv)
    if is_debug:
        print("Inverting", Z.shape)
    if is_woodbury:
        z_transpose = Z.T
        small_inv = np.linalg.inv(z_transpose.dot(Z) + lamda * np.identity(Z.shape[1]))
        tmp = Z.dot(small_inv.dot(z_transpose.dot(solution_mat)))
        optimal = (1. / lamda) * (solution_mat - tmp)
    else:
        preconditioner = Z.dot(Z.T) + lamda * np.identity(Z.shape[0])
        preconditioner_inv = np.linalg.inv(preconditioner)
        optimal = preconditioner_inv.dot(solution_mat)
    if is_debug:
        print("Predicting solution")
    total = [0] * len(validation_data)
    selected = data_mat[indexes]
    check = np.transpose(optimal).dot(Z)
    for i, single_validation_data in enumerate(validation_data):
        converted_validation_data = create_converted_nystrom(single_validation_data, r_inv, selected, sigma)
        res_num = check.dot(np.transpose(converted_validation_data))
        total[i] = res_num
    solution = np.transpose(np.array(total))
    if is_debug:
        print (solution)
    return solution


def prepare_solution_mat_for_one_vs_many(current_group, solution_mat):
    res = []
    for single_solution in solution_mat:
        res.append(np.array(1 if single_solution == current_group else -1))
    return np.transpose(np.array(res))


def test_sketching_method(data_mat, validation_data, solution_mat, validation_solution,
                          is_classification, sigma, lamda, is_woodbury, s, is_nystrom, is_debug = False):
    try:
        if not is_classification:
            if is_debug:
                print "Preforming a regression"
            if is_nystrom:
                prediction = create_nystrom_preconfitioner_inner(data_mat, is_debug, is_woodbury,
                                                                 lamda, s, sigma, solution_mat, validation_data)
            else:
                prediction = create_fourrier_preconditioner_inner(data_mat, is_debug, is_woodbury,
                                                                  lamda, s, sigma, solution_mat, validation_data)
            diff = np.linalg.norm(prediction - np.transpose(validation_solution), 2) / np.linalg.norm(np.transpose(validation_solution), 2)
            return diff
        else:
            possible_groups = np.unique(validation_data)
            if is_debug:
                print "Preforming a classification", len(possible_groups)
            all_results = collections.defaultdict(dict)
            if len(possible_groups) > 2:
                for k, current_group in enumerate(possible_groups):
                    before = time.time()
                    solution_mat_copy = prepare_solution_mat_for_one_vs_many(current_group, solution_mat)
                    if is_nystrom:
                        prediction = create_nystrom_preconfitioner_inner(data_mat, is_debug, is_woodbury, lamda, s, sigma,
                                                                         solution_mat_copy, validation_data)
                    else:
                        prediction = create_fourrier_preconditioner_inner(data_mat, is_debug, is_woodbury, lamda, s, sigma,
                                                                          solution_mat_copy, validation_data)
                    for i, data in enumerate(validation_data):
                        all_results[i][current_group] = prediction[i]

                    print float(k) / len(possible_groups), time.time() - before
            else:
                solution_mat_copy = prepare_solution_mat_for_one_vs_many(1, solution_mat)
                if is_nystrom:
                    prediction = create_nystrom_preconfitioner_inner(data_mat, is_debug, is_woodbury, lamda, s, sigma,
                                                                     solution_mat_copy, validation_data)
                else:
                    prediction = create_fourrier_preconditioner_inner(data_mat, is_debug, is_woodbury, lamda, s, sigma,
                                                                      solution_mat_copy,
                                                                      validation_data)
                for i, data in enumerate(validation_data):
                    all_results[i][1 if prediction[i] > 0 else -1] = prediction[i]

            concluded_results = np.array([0] * len(all_results))
            for data_idx, all_results in all_results.iteritems():
                concluded_results[data_idx] = max(all_results.items(), key=lambda x:x[1])[0]

            count = 0
            for i in range(len(concluded_results)):
                if concluded_results[i] != validation_solution[i][0]:
                    count += 1
            score = float(count) * 100 / len(concluded_results)
            return score
    except Exception, e:
        if is_debug:
            raise
        return -1


def perform_cross_validation(data_mat, validation_mat, learn_limit, validation_limit,
                             is_classification, possible_sigma, possible_s, is_nystrom, is_debug = False, iter_num = 1):
    sigma_scores = {}
    for s in possible_s:
        for sigma in possible_sigma:
            sigma_score_sum = 0
            tests = 0
            if is_debug:
                print "sigma, ", sigma, s
            learn_limit = max(s * 2, learn_limit)
            random_indexes = np.random.choice(range(data_mat.shape[0]), learn_limit + validation_limit, replace=False)
            learn_indexes = random_indexes[:learn_limit]
            validation_indexes = random_indexes[learn_limit:]

            learn_data = data_mat[learn_indexes]
            learn_solutions = validation_mat[learn_indexes]

            validation_data = data_mat[validation_indexes]
            validation_solutions = validation_mat[validation_indexes]

            for i in range(iter_num):
                if is_debug:
                    print "iter", i, learn_data.shape, validation_data.shape
                res = test_sketching_method(learn_data, validation_data, learn_solutions, validation_solutions,
                                      is_classification, sigma, lamda, True, s, is_nystrom, is_debug = is_debug)
                if res != -1:
                    sigma_score_sum += res
                    tests += 1
            sigma_scores[(sigma, s)] = sigma_score_sum / tests if tests != 0 else 100

    return min(sigma_scores.items(), key=lambda x: x[1])[0]

if __name__ == '__main__':
    # mat = scipy.io.loadmat(r'C:\Users\USER\Downloads\data_q3\adult.mat')
    # sigma, s = 4.0, 500
    # lamda = mat["lambda"][0][0]
    # is_classification = "L" in mat
    # is_nystrom = True
    #
    # solution_data = mat["L"] if is_classification else mat["Y"]
    # validation_data = mat["Lt"] if is_classification else mat["Yt"]
    # before = time.time()
    # res = test_sketching_method(mat["X"][:10000], mat["Xt"][:10000], solution_data[:10000], validation_data[:10000],
    #                             is_classification, sigma, lamda, is_classification, s, is_nystrom, is_debug=True)
    # print "Took:", time.time() - before, res

    #FILE_NAMES = ["acoustic", "adult", "cadata", "cod-rna", "cpu", "ijcnn1", "mnist"]
    FILE_NAMES = ["cod-rna", "cpu", "mnist"]
    FILE_NAMES = ["adult"]

    for name in FILE_NAMES:
        print "Running: ", name
        mat = scipy.io.loadmat(r'C:\Users\USER\Downloads\data_q3\%s.mat' % name)
        s = 30
        lamda = mat["lambda"][0][0]
        sigma = mat["sigma"][0][0]
        is_classification = "L" in mat
        is_nystrom = True

        solution_data = mat["L"] if is_classification else mat["Y"]
        validation_data = mat["Lt"] if is_classification else mat["Yt"]

        possibilities = np.concatenate((np.arange(0.5, 1, 0.5) * sigma, np.arange(1, 10, 3) * sigma))
        possibilities_s = np.array([300])

        cross_validation_learning = 500
        cross_validation_validation = 100

        is_debug = True
        print mat["X"][:10]

        # sigma, s = perform_cross_validation(mat["X"], solution_data, cross_validation_learning, cross_validation_validation, is_classification, possibilities,
        #                                     possibilities_s, is_nystrom, is_debug)
        sigma, s = 2.5, 10
        # print "optimal params", sigma, s
        # print mat["X"][range(10)]
        # res = test_sketching_method(mat["X"], mat["Xt"], solution_data, validation_data,
        #                             is_classification, sigma, lamda, True, s, is_nystrom, is_debug=is_debug)
        # print "Nystrom ", name, "optimal params", sigma, s, "result: ", res
        #
        is_nystrom = False
        # sigma, s = perform_cross_validation(mat["X"], solution_data, cross_validation_learning, cross_validation_validation, is_classification, possibilities,
        #                                     possibilities_s, is_nystrom, is_debug)
        print "optimal params", sigma, s
        res = test_sketching_method(mat["X"], mat["Xt"], solution_data, validation_data,
                                    is_classification, sigma, lamda, True, s, is_nystrom, is_debug=is_debug)
        print "Fourrier ", name, "optimal params", sigma, s, "result: ", res
