#from models_script import RegEffectModel
import numpy as np
import pandas as pd


# ---------------------------------------- Evaluation: RMSE ----------------------------------------


def RMSE(pred_val, true_val):

    return np.sqrt(mean_squared_error(pred_val, true_val))


# ---------------------------------------- RegEffectModel ----------------------------------------


class RegEffectModel():
    """
    @param damping_term: (int) regularized term
    """

    def __init__(self, damping_term):
        self.damping_term = damping_term

    def fit(self, X_train):
        """
        @param X_train: (dataframe) the training set of movie ratings from users

        """

        X_train = X_train[['user_id', 'movie_id', 'rating']].copy()

        # overall_mean of ratings
        self.mu = X_train['rating'].mean()

        # bu
        I_u = X_train.groupby('user_id')['rating'].count()
        b_u = 1/(I_u + self.damping_term) * \
            (X_train.groupby('user_id')['rating'].sum() - I_u * self.mu)
        self.b_u = b_u.rename('b_u')

        # bi
        X_train = X_train.join(self.b_u, on='user_id')
        X_train['dev_i'] = X_train['rating'] - X_train['b_u'] - self.mu
        U_i = X_train.groupby('movie_id')['rating'].count()
        b_i = 1/(U_i + self.damping_term) * (X_train.groupby('movie_id')['dev_i'].sum())
        self.b_i = b_i.rename('b_i')

    def predict(self, X_test):
        """
        @param X_test: (dataframe) the testing set of movie ratings from users

        @return: (1-d array) the prediction of testing set
        """

        X_test = X_test[['user_id', 'movie_id']].copy()

        # join b_u and b_i for testing set
        X_test = X_test.join(self.b_u, on='user_id')
        X_test = X_test.join(self.b_i, on='movie_id')
        X_test = X_test.fillna(self.mu)

        return (self.mu + X_test['b_u'] + X_test['b_i']).values


# ---------------------------------------- SGD ----------------------------------------


class SGD():
    """
    @param n_factors: (int) the number of latent factors in rating matrix
    @param n_itr: (int) the number of iterations for SGD
    @param learning_rate: (float) the learning step in SGD
    @param reg_bu: (float) the regulartion term of user bias
    @param reg_bi: (float) the regulartion term of movie bias
    @param reg_Pu: (float) the regulartion term of user matrix
    @param reg_Qi: (float) the regulartion term of movie matrix
    @param bias_effect: (boolean) if ture: the initial values of b_u and b_i from RegEffectModel

    """

    def __init__(self, n_factors=5, n_itr=5, learning_rate=0.01, reg_bu=0.0, reg_bi=0.0,
                 reg_Pu=0.0, reg_Qi=0.0, bias_effect=True):

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_itr = n_itr
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.reg_Pu = reg_Pu
        self.reg_Qi = reg_Qi
        self.bias_effect = bias_effect

    def _initial_param(self, X_train):

        # initial rating matrix
        self.R = pd.pivot_table(data=X_train,
                                values='rating',
                                index='user_id',
                                columns='movie_id').fillna(0).values

        # define mapping relationship dict{old_index : new_index}
        self.user_map = self.map_id(X_train, 'user_id')
        self.movie_map = self.map_id(X_train, 'movie_id')

        # get number of users/movies from Rating Matrix
        self.n_users = self.R.shape[0]
        self.n_movies = self.R.shape[1]

        # initial Pu user matrix
        self.P_u = np.random.normal(loc=0, scale=1/self.n_factors,
                                    size=(self.n_users, self.n_factors))
        # initial Qi user matrix
        self.Q_i = np.random.normal(loc=0, scale=1/self.n_factors,
                                    size=(self.n_movies, self.n_factors))

        # initial user bias, movie bias term, overall mean
        if self.bias_effect == True:
            self.baseline_algo = RegEffectModel(damping_term=25)
            self.baseline_algo.fit(X_train)
            self.b_u, self.b_i = self.baseline_algo.b_u.values, self.baseline_algo.b_i.values
        else:
            self.b_u = np.zeros(self.n_users)
            self.b_i = np.zeros(self.n_movies)

        self.u = X_train['rating'].mean()

    def fit(self, X_train):
        """
        @param X_train: (dataframe) the training set of movie ratings from users

        """
        X_train = X_train.copy()
        self._initial_param(X_train)

        X_train['user_id'] = X_train['user_id'].map(self.user_map)
        X_train['movie_id'] = X_train['movie_id'].map(self.movie_map)

        itr = 0
        while itr < self.n_itr:

            # Generate random indexes for SGD
            index = np.arange(len(X_train))
            np.random.shuffle(index)

            # Loop for each r_ui in Rating Matrix
            for row in X_train.iloc[index].itertuples():

                index, user_index, movie_index, rating = row[:4]
                # calculate e_ui = r_ui - est_r_ui
                self.err = self.R[user_index, movie_index] - self.pred_r_ui(user_index, movie_index)
                # update parameters
                self.b_u[user_index] += self.learning_rate * \
                    (self.err - self.reg_bu * self.b_u[user_index])
                self.b_i[movie_index] += self.learning_rate * \
                    (self.err - self.reg_bi * self.b_i[movie_index])
                self.P_u[user_index, :] += self.learning_rate * \
                    (self.err * self.Q_i[movie_index, :] - self.reg_Pu * self.P_u[user_index, :])
                self.Q_i[movie_index, :] += self.learning_rate * \
                    (self.err * self.P_u[user_index, :] - self.reg_Qi * self.Q_i[movie_index, :])

            itr += 1

    def predict(self, X_test):
        """
        @param X_test: (dataframe) the testing set of movie ratings from users

        @return: (1-d array) the prediction of testing set
        """

        X_test = X_test.iloc[:, :3].copy()

        # find the mapping indexes
        check_id = X_test['user_id'].isin(
            self.user_map.keys()) & X_test['movie_id'].isin(self.movie_map.keys())

        # classify known-dataset and unknown-dataset
        X_known, X_unknown = X_test[check_id], X_test[-check_id]
        # by using map, find out new user/movie indexes
        user_inds = X_known['user_id'].map(self.user_map)
        movie_inds = X_known['movie_id'].map(self.movie_map)

        # make prediction
        rating_pred = np.array([
            self.pred_r_ui(u_ind, i_ind)
            for u_ind, i_ind in zip(user_inds, movie_inds)
        ])

        # assign predictions to original dataframe
        X_test.loc[check_id, 'rating'] = rating_pred

        if self.bias_effect == True:
            X_test.loc[-check_id, 'rating'] = self.baseline_algo.predict(X_unknown)
        else:
            X_test.loc[-check_id, 'rating'] = self.u

        return X_test['rating'].values

    def pred_r_ui(self, user, movie):
        """
        @user(int): user index
        @movie(int):  movie index

        @return(float):  prediction value
        """

        pred = self.u + self.b_u[user] + self.b_i[movie] + \
            np.dot(self.P_u[user, :], self.Q_i[movie, :])
        return pred

    def map_id(self, X, column_name):
        """
        @X(pd.dataframe): traning dataset
        @column_name(str): indicate column name for mapping

        @return(dict): return mapping relationships in dictionary
        """

        old_ids = np.unique(X[column_name])
        new_ids = np.arange(X[column_name].nunique())

        return dict(zip(old_ids, new_ids))


# ---------------------------------------- ALS ----------------------------------------

class ALS():

    def __init__(self, n_factors=5, n_itr=5, reg_Pu=0.0, reg_Qi=0.0, random_seed=0,
                 bias_effect=True):
        """
        @param n_factors: (int) the number of latent factors in rating matrix
        @param n_itr: (int) the number of iterations
        @param reg_Pu: (float) the regularization term of User Matrix
        @param reg_Qi: (float) the regularization term of Movie Matrix
        @param random_seed: (int) set random seed for initlization of User/Movie Matrix
        @bias_effect: (boolean) whether use the baseline model to predict missing values in testing set

        """

        self.n_factors = n_factors
        self.n_itr = n_itr
        self.reg_Pu = reg_Pu
        self.reg_Qi = reg_Qi
        self.random_seed = random_seed
        self.bias_effect = bias_effect

    def _initial_param(self, X_train):

        X_train = X_train.copy()

        # initial rating matrix
        self.R = pd.pivot_table(data=X_train,
                                values='rating',
                                index='user_id',
                                columns='movie_id').fillna(0).values

        # define mapping relationship dict{old_index : new_index}
        self.user_map = self.map_id(X_train, 'user_id')
        self.movie_map = self.map_id(X_train, 'movie_id')

        # get number of users/movies from Rating Matrix
        self.n_users = self.R.shape[0]
        self.n_movies = self.R.shape[1]

        np.random.seed(self.random_seed)
        # initial P_u, Q_i, and the identity matrix
        self.P_u = 3 * np.random.rand(self.n_factors, self.n_users)  # k*n
        self.Q_i = 3 * np.random.rand(self.n_factors, self.n_movies)  # k*m
        self.I = np.identity(self.n_factors)  # k*k

        # The goal is to fillin the missing ratings if user_id or movie_id in training set
        # but not in testing set.
        # initial user bias, movie bias term, overall mean
        if self.bias_effect == True:
            self.baseline_algo = RegEffectModel(damping_term=25)
            self.baseline_algo.fit(X_train)
            self.b_u, self.b_i = self.baseline_algo.b_u.values, self.baseline_algo.b_i.values
        else:
            self.b_u = np.zeros(self.n_users)
            self.b_i = np.zeros(self.n_movies)

        self.u = X_train['rating'].mean()

    def fit(self, X_train):

        X_train = X_train.copy()
        self._initial_param(X_train)

        itr = 0
        while itr < self.n_itr:

            ########## Fix Q_i, Update P_u ##########
            # loop for each user
            for i, Ri in enumerate(self.R):

                # Determine the scale of penalities, lambdau, for each user

                # number of items user i has rated
                reg_scale_i = np.count_nonzero(Ri)
                # if no ratings in Rating Matrix
                if reg_scale_i == 0:
                    reg_scale_i = 1

                # Find indicies
                # get the array of nonzero indicies in row i for each user
                Ri_nonzero_idx = np.nonzero(Ri)[0]
                # select subset of row R associated with movies reviewd by user i -> find r_ui
                r_ui = self.R[i, Ri_nonzero_idx]
                # select subset of Q_i associated with movies by user i -> find qi_T
                qi = self.Q_i[:, Ri_nonzero_idx]

                # Compute
                # qi * ri
                bi = qi.dot(r_ui)
                # qi* qi.T + lambda_u * I
                Ai = qi.dot(qi.T) + self.reg_Pu * reg_scale_i * self.I

                # get p_ui
                self.P_u[:, i] = np.linalg.solve(Ai, bi)  # bi.dot(np.linalg.inv(Ai))

            ########## Fix P_u, Update Q_i ##########
            # loop for each user
            for j, Rj in enumerate(self.R.T):

                # Determine the scale of penalities, lambdau, for each user

                # number of items user i has rated
                reg_scale_j = np.count_nonzero(Rj)
                # if no ratings in Rating Matrix
                if reg_scale_j == 0:
                    reg_scale_j = 1

                # Find indicies
                # get the array of nonzero indicies in row i for each movie
                Rj_nonzero_idx = np.nonzero(Rj)[0]
                # select subset of row R associated with users who reviewed movie j -> find r_ui
                r_uj = self.R.T[j, Rj_nonzero_idx]
                # select subset of Q_i associated with users who reviewed movie j -> find pu
                pu = self.P_u[:, Rj_nonzero_idx]

                # Compute
                # qi * ri
                bj = pu.dot(r_uj)
                # qi* qi.T + lambda_u * I
                Aj = pu.dot(pu.T) + self.reg_Qi * reg_scale_j * self.I

                # get q_ui
                self.Q_i[:, j] = np.linalg.solve(Aj, bj)  # bj.dot(np.linalg.inv(Aj))

            itr += 1

    def predict(self, X_test):

        X_test = X_test.iloc[:, :3].copy()

        # find the mapping indexes
        check_id = X_test['user_id'].isin(
            self.user_map.keys()) & X_test['movie_id'].isin(self.movie_map.keys())

        # classify known-dataset and unknown-dataset
        X_known, X_unknown = X_test[check_id], X_test[-check_id]
        # by using map, find out new user/movie indexes
        user_inds = X_known['user_id'].map(self.user_map)
        movie_inds = X_known['movie_id'].map(self.movie_map)

        # make prediction
        rating_pred = np.array([
            self.P_u[:, u_ind].dot(self.Q_i[:, i_ind])
            for u_ind, i_ind in zip(user_inds, movie_inds)
        ])

        # assign predictions to original dataframe
        X_test.loc[check_id, 'rating'] = rating_pred

        # fillout missing values of ratings whose use_id or movie_id is not in trianing set
        if self.bias_effect == True:
            X_test.loc[-check_id, 'rating'] = self.baseline_algo.predict(X_unknown)
        else:
            X_test.loc[-check_id, 'rating'] = self.u

        # deal with some outliers from rating_pred
        min_rating = np.min(self.R[np.nonzero(self.R)])
        max_rating = np.max(self.R)
        X_test.loc[X_test['rating'] < min_rating, 'rating'] = min_rating
        X_test.loc[X_test['rating'] > max_rating, 'rating'] = max_rating

        return X_test['rating'].values

    def map_id(self, X, column_name):
        """
        @X(pd.dataframe): traning dataset
        @column_name(str): indicate column name for mapping

        @return(dict): return mapping relationships in dictionary
        """

        old_ids = np.unique(X[column_name])
        new_ids = np.arange(X[column_name].nunique())

        return dict(zip(old_ids, new_ids))
