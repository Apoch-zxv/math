#define ARMA_USE_LAPACK

#include <iostream>
#include <matio.h>
#include <armadillo>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

using namespace std;
using namespace arma;


void copy_single_mat(matvar_t *raw_mat, mat& res) {
    res = mat(raw_mat->dims[0], raw_mat->dims[1]);
    if (raw_mat->class_type == 6) {
        const double *data = static_cast<const double *>(raw_mat->data);

        for (size_t j = 0; j < raw_mat->dims[1]; j++) {
            for (size_t i = 0; i < raw_mat->dims[0]; i++) {
                res(i, j) = data[j * raw_mat->dims[0] + i];
            }
        }
    } else if (raw_mat->class_type == 10) {
        const int16_t *data = static_cast<const int16_t *>(raw_mat->data);

        for (size_t j = 0; j < raw_mat->dims[1]; j++) {
            for (size_t i = 0; i < raw_mat->dims[0]; i++) {
                res(i, j) = data[j * raw_mat->dims[0] + i];
            }
        }
    } else if (raw_mat->class_type == 9) {
        const uint8_t *data = static_cast<const uint8_t *>(raw_mat->data);

        for (size_t j = 0; j < raw_mat->dims[1]; j++) {
            for (size_t i = 0; i < raw_mat->dims[0]; i++) {
                res(i, j) = data[j * raw_mat->dims[0] + i];
            }
        }
    } else {
        exit(5);
    }
}

void read_file(const char* name, mat& data, mat& solutions,
              mat& validation_data, mat& validation_data_solutions,
              bool& is_classification, double& lambda, double& sigma) {
    char filename[100];
    sprintf(filename, "/home/grg/Downloads/%s.mat", name);

    mat_t *mat_file = Mat_Open(filename, MAT_ACC_RDONLY);
    matvar_t *raw_mat = Mat_VarRead(mat_file, (char *) "X");
    copy_single_mat(raw_mat, data);

    raw_mat = Mat_VarRead(mat_file, (char *) "Xt");
    copy_single_mat(raw_mat, validation_data);

    matvar_t *raw_mat_validation_solution;
    raw_mat = Mat_VarRead(mat_file, (char *) "Y");
    if (raw_mat == NULL) {
        is_classification = true;
        raw_mat = Mat_VarRead(mat_file, (char *) "L");
        raw_mat_validation_solution = Mat_VarRead(mat_file, (char *) "Lt");
    } else {
        is_classification = false;
        raw_mat_validation_solution = Mat_VarRead(mat_file, (char *) "Yt");
    }

    copy_single_mat(raw_mat, solutions);
    copy_single_mat(raw_mat_validation_solution, validation_data_solutions);

    matvar_t *raw_sigma = Mat_VarRead(mat_file, (char *) "sigma");
    if (raw_sigma->data_type == 9) {
        const double *raw_sigma_data = static_cast<const double *>(raw_sigma->data);
        sigma = *raw_sigma_data;
    } else if(raw_sigma->data_type == 2) {
        const uint8_t *raw_sigma_data = static_cast<const uint8_t *>(raw_sigma->data);
        sigma = *raw_sigma_data;
    }

    matvar_t *raw_lambda = Mat_VarRead(mat_file, (char*)"lambda") ;
    const double *raw_lambda_data = static_cast<const double*>(raw_lambda->data) ;
    lambda = *raw_lambda_data;
}

double gaussian_kernel(const rowvec& row1, const rowvec& row2, double sigma) {
    double norm = arma::norm(row1 - row2, 2);
    double inner = (-1 * norm * norm) / (2 * sigma * sigma);
    double res = exp(inner);
    return res;
}


void create_converted(mat& dest, mat& data, const rowvec& row, vector<unsigned int> indices, mat& inv, double &sigma){
    mat single_res(1, indices.size());
    for (size_t i = 0; i < indices.size(); i ++) {
        single_res(0, i) = gaussian_kernel(row, data.row(indices[i]), sigma);
    }
    dest = single_res * inv;
}

bool perform_fourrier_regression(mat &dest, mat& data, mat& solutions,
                                mat& validation_data, mat& validation_data_solutions,
                                bool& is_classification, double lambda, double sigma, int s) {
    mat random_ws = randn(s, data.n_cols) * (1. / (sigma * sigma));
    mat random_bs = randu(1, s) * 2 * 3.1415;

//    cout << "Building Z" << endl;
    double sqrt_2 = sqrt(2);
    double sqrt_s = 1. / sqrt(s);
    mat Z(data.n_rows, s);
    for (size_t i = 0; i < data.n_rows; i ++) {
        mat applied = trans(random_ws * trans(data.row(i)));
        for (size_t j = 0; j < s; j ++) {
            Z(i, j) = cos(applied(0, j) + random_bs(0, j)) * sqrt_2 * sqrt_s;
        }
    }

//    cout << "Inverting" << endl;
    mat Z_trans = trans(Z);
    mat I(Z.n_cols, Z.n_cols, fill::eye);
    mat small_inv = inv(Z_trans * Z + lambda * I);
    mat tmp = Z * (small_inv * (Z_trans * solutions));
    mat optimal = (1. / lambda) * (solutions - tmp);
    mat optimal_trans = trans(optimal);

//    cout << "Predicting" << endl;
    mat prediction_vec(1, validation_data.n_rows);
    mat check = optimal_trans * Z;
    for (size_t i = 0; i < validation_data.n_rows; i ++ ) {
        mat converted(1, s);
        mat applied = trans(random_ws * trans(validation_data.row(i)));
        for (size_t j = 0; j < s; j ++) {
            converted(0, j) = cos(applied(0, j) + random_bs(0, j)) * sqrt_2 * sqrt_s;
        }

        mat res = (check * trans(converted));
        double prediction = res.at(0, 0);
        prediction_vec(0, i) = prediction;
    }
    dest = trans(prediction_vec);
    return true;
}


bool perform_nystrom_regression(mat &dest, mat& data, mat& solutions,
                          mat& validation_data, mat& validation_data_solutions,
                          bool& is_classification, double lambda, double sigma, int s, int max_nystrom_iter) {
    std::srand ( unsigned ( std::time(0) ) );
    mat r;
    vector<unsigned int> indices(data.n_rows);
    for (size_t iter = 0; iter < max_nystrom_iter; iter++) {
        try {
            iota(indices.begin(), indices.end(), 0);
            random_shuffle(indices.begin(), indices.end());

            indices = vector<unsigned int>(indices.begin(), indices.begin() + s);

            //    cout << "Matrix size: " << data.n_rows << " " << data.n_cols << endl;
            mat middle_mat(s, s);
            for (size_t i = 0; i < s; i++) {
                unsigned int index1 = indices[i];
                for (size_t j = 0; j < s; j++) {
                    unsigned int index2 = indices[j];
                    middle_mat(i, j) = gaussian_kernel(data.row(index1), data.row(index2), sigma);
                }
            }

            //    cout << "Decomposing" << endl;
            r = chol(middle_mat);
            break;
        } catch (...) { /* */
            if (iter == max_nystrom_iter - 1) {
                return false;
            }
        }
    }
    mat r_inv = inv(r);

//    cout << "Creating left mat" << endl;
    mat left_mat(s, data.n_rows);
    for (size_t i=0; i < s; i ++) {
        unsigned int index1 = indices[i];
        for (size_t j=0; j < data.n_rows; j ++) {
            left_mat(i, j) = gaussian_kernel(data.row(index1), data.row(j), sigma);
        }
    }
    left_mat = trans(left_mat);
//    cout << "Inverting" << endl;
    mat Z = left_mat * r_inv;
    mat Z_trans = trans(Z);
    mat I(Z.n_cols, Z.n_cols, fill::eye);
    mat small_inv = inv(Z_trans * Z + lambda * I);
    mat tmp = Z * (small_inv * (Z_trans * solutions));
    mat optimal = (1. / lambda) * (solutions - tmp);

    mat optimal_trans = trans(optimal);
    mat prediction_vec(1, validation_data.n_rows);
//    cout << "Calc prediction" << endl;
    mat check = optimal_trans * Z;
    for (size_t i = 0; i < validation_data.n_rows; i ++ ) {
        mat converted;
        create_converted(converted, data, validation_data.row(i), indices, r_inv, sigma);
        mat res = (check * trans(converted));
        double prediction = res.at(0, 0);
        prediction_vec(0, i) = prediction;
    }
    dest = trans(prediction_vec);
    return true;
}

mat copy_index_rows(mat& source, vector<unsigned int> indices) {
    mat dest(indices.size(), source.n_cols);
    for (size_t i = 0; i < indices.size(); i ++) {
        for (size_t j = 0; j < source.n_cols; j ++) {
            dest(i, j) = source(indices[i], j);
        }
    }
    return dest;
}

mat prepare_for_one_vs_many(int current_label, mat& data) {
    mat res(data.n_rows, data.n_cols);
    for (size_t i = 0; i < data.n_rows; i ++) {
        for (size_t j = 0; j < data.n_cols; j ++) {
            if (round(data(i, j)) == current_label) {
                res(i, j) = 1;
            } else {
                res(i, j) = -1;
            }
        }
    }
    return res;
}

double fit_predict(mat& data, mat& solutions,
                   mat& validation_data, mat& validation_data_solutions,
                   bool& is_classification, double lambda, double sigma, int s, bool is_nystrom, mat& unique_labels, int max_nystrom_iter) {
    mat dest;
    if (!is_classification) {
        bool res;
        if (is_nystrom) {
            res = perform_nystrom_regression(dest, data, solutions, validation_data, validation_data_solutions,
                                       is_classification, lambda, sigma, s, max_nystrom_iter);
        } else {
            res = perform_fourrier_regression(dest, data, solutions, validation_data, validation_data_solutions,
                                       is_classification, lambda, sigma, s);
        }
        if (res) {
            return arma::norm(dest - validation_data_solutions, 2) / arma::norm(validation_data_solutions, 2);
        } else {
            return 1000;
        }
    } else {
        bool res;
        mat predictions(validation_data.n_rows, unique_labels.n_rows);
        for (size_t label_row = 0; label_row < unique_labels.n_rows; label_row ++) {
            int current_label = round(unique_labels(label_row, 0));
            mat one_vs_many_solutions = prepare_for_one_vs_many(current_label, solutions);

            if (is_nystrom) {
                res = perform_nystrom_regression(dest, data, one_vs_many_solutions, validation_data, validation_data_solutions,
                                           is_classification, lambda, sigma, s, max_nystrom_iter);
            } else {
                res = perform_fourrier_regression(dest, data, one_vs_many_solutions, validation_data, validation_data_solutions,
                                            is_classification, lambda, sigma, s);
            }

            if (!res) {
                return 1000;
            }

            //cout << "Summing predictions: " << validation_data.n_rows << endl;
            for (size_t solution_idx = 0; solution_idx < validation_data.n_rows; solution_idx ++) {
                predictions(solution_idx, label_row) = dest(solution_idx, 0);
            }
        }

        int count_error = 0;
        for (size_t solution_idx = 0; solution_idx < validation_data.n_rows; solution_idx ++) {
            double max_prediction = -1000;
            int max_prediction_label = 0;
            for (size_t label_row = 0; label_row < unique_labels.n_rows; label_row ++) {
                if (predictions(solution_idx, label_row) > max_prediction) {
                    max_prediction = predictions(solution_idx, label_row);
                    max_prediction_label = round(unique_labels(label_row, 0));
                }
            }

            if (round(validation_data_solutions(solution_idx, 0)) != max_prediction_label) {
                count_error ++;
            }
        }
        return (count_error / (double) validation_data.n_rows);
    }
}

double cross_validate_sigma(unsigned int cross_validation_learn_size, unsigned int cross_validation_size,
                          mat& data, mat& solutions,
                          bool& is_classification, double lambda, double sigma, int s, bool is_nystrom, int max_iter) {
    mat u;
    if (is_classification) {
        u = unique(solutions);
    }

    double best_score = 1000;
    double best_sigma = sigma;
    for (double factor = 0.1; factor < 5; factor += 0.05) {
        double current_sigma = factor * sigma;
        double avg_score = 0;
        for (size_t iter = 0; iter < max_iter; iter ++) {
            vector<unsigned int> indices(data.n_rows);
            iota(indices.begin(), indices.end(), 0);
            random_shuffle(indices.begin(), indices.end());

            vector<unsigned int> learning_indices(indices.begin(), indices.begin() + cross_validation_learn_size);
            vector<unsigned int> validation_indices(indices.begin() + cross_validation_learn_size,
                                                    indices.begin() + cross_validation_learn_size + cross_validation_size);

            mat cross_validation_learning_mat = copy_index_rows(data, learning_indices);
            mat cross_validation_learing_solution_mat = copy_index_rows(solutions, learning_indices);

            mat cross_validation_validation_mat = copy_index_rows(data, validation_indices);
            mat cross_validation_validation_solution_mat = copy_index_rows(solutions,
                                                                           validation_indices);

            double score = fit_predict(cross_validation_learning_mat, cross_validation_learing_solution_mat,
                                       cross_validation_validation_mat, cross_validation_validation_solution_mat,
                                       is_classification, lambda, current_sigma, s, is_nystrom, u, 50);
            if (score > 10) {
                score = avg_score;
            }
            size_t trials = iter + 1;
            avg_score = (avg_score * (trials - 1) + score) / trials;
        }
        if (avg_score < best_score) {
            best_score = avg_score;
            best_sigma = current_sigma;
        }
    }
    return best_sigma;
}

void analyze_file(const char* name, size_t s, int cross_validation_repeat, int max_nystrom_iter, bool is_run_fourrier = true, bool is_run_nystrom = true) {
    mat dest;
    mat data;
    mat solutions;
    mat validation_data;
    mat validation_data_solutions;
    bool is_classification;
    double lambda;
    double sigma;
    read_file(name, data, solutions, validation_data, validation_data_solutions, is_classification, lambda, sigma);
    //read_file("ijcnn1.mat_nosparse", data, solutions, validation_data, validation_data_solutions, is_classification, lambda, sigma);
    dest = unique(solutions);

    if (is_run_fourrier) {
        double fourrier_optimal_sigma = cross_validate_sigma(2000, 1000, data, solutions, is_classification, lambda,
                                                             sigma, s, false, cross_validation_repeat);
        cout << name << " Fourrier sigma: " << fourrier_optimal_sigma << endl;
        double score = fit_predict(data, solutions, validation_data, validation_data_solutions, is_classification,
                                   lambda, fourrier_optimal_sigma, s, false, dest, max_nystrom_iter);
        cout << name << " Fourrier score: " << score << endl;
    }


    if (is_run_nystrom) {
        double nystrom_optimal_sigma = cross_validate_sigma(2000, 1000, data, solutions, is_classification, lambda,
                                                            sigma, s, true, cross_validation_repeat);
        cout << name << " Nystrom sigma: " << nystrom_optimal_sigma << endl;
        double score1 = fit_predict(data, solutions, validation_data, validation_data_solutions, is_classification,
                                    lambda, nystrom_optimal_sigma, s, true, dest, max_nystrom_iter);
        cout << name << " Nystrom score: " << score1 << endl;
    }
}


int main(const char* name) {
    size_t s = 300;
    cout << "***************************************************" << endl;
    cout << s << endl;
    //analyze_file("acoustic.mat_nosparse", s, 5);
    //analyze_file("adult", s, 5, false);
    //analyze_file("cadata", s, 5);
    //analyze_file("cod-rna", s, 5);
    //analyze_file("cpu", s, 5);
    //analyze_file("ijcnn1.mat_nosparse", s, 5);
    analyze_file("mnist", s, 5, 1000000);

    s = 600;
    cout << "***************************************************" << endl;
    cout << s << endl;
    //analyze_file("acoustic.mat_nosparse", s, 5);
    //analyze_file("adult", s, 5, false, false);
    //analyze_file("cadata", s, 5);
    //analyze_file("cod-rna", s, 5, false);
    //analyze_file("cpu", s, 5);
    //analyze_file("ijcnn1.mat_nosparse", s, 5);
    analyze_file("mnist", s, 5, 1000000);

/*    s = 1000;
    cout << "***************************************************" << endl;
    cout << s << endl;
    analyze_file("acoustic.mat_nosparse", s, 5);
    analyze_file("adult", s, 5);
    analyze_file("cadata", s, 5);
    analyze_file("cod-rna", s, 5);
    analyze_file("cpu", s, 5);
    analyze_file("ijcnn1.mat_nosparse", s, 5);*/
    //analyze_file("mnist", s, 5);
}