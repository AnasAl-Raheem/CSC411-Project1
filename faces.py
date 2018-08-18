import os
import numpy as np
from numpy.linalg import norm
from scipy.misc import imread
import matplotlib.pyplot as plt
import shutil

# cropped images location
actors_folder = "cropped_male/"
actresses_folder = "cropped_female/"
figures_folder = "figures/"

# problemtic images, to be ignored
discard_list = ["bracco87.jpg", "chenoweth71.jpg", "drescher69.jpg", "drescher81.jpg",
                "hader11.jpg", "hader95.jpg",
                "vartan13.jpg", "vartan14.jpg", "vartan78.jpg", "vartan116.jpg"]


# represents part2, upload images as numpy arrays. return a dictionary with act names as keys, each key has a list of 3 lists
# each represent test, validation and training sets.
def get_images(src_dir, names, test_size, validation_size, training_size):
    files = sorted(os.listdir(src_dir))
    act_dict = {i: [j for j in files if j.startswith(i)] for i in names}
    for i in act_dict.keys():
        files_names = act_dict[i]
        files = []
        for j in files_names:
            if not j in discard_list:
                im = imread(src_dir + j, "L")
                files.append(np.append([1], np.ndarray.flatten(im) / 255.0))
            else:
                print "discard " + j
        np.random.seed(0)
        np.random.shuffle(files)
        np.random.shuffle(files)
        np.random.shuffle(files)
        act_dict[i] = [files[0:test_size], files[test_size:test_size + validation_size],
                       files[test_size + validation_size:min(test_size + validation_size + training_size, len(files))]]
    return act_dict

def f(x, y, theta):
    return np.sum((np.dot(theta.T, x) - y)**2)

def df(x, y, theta):
    return 2. * np.dot(x, (np.dot(theta.T, x) - y))

def grad_descent(f, df, x, y, init_t, alpha, max_iter = 30000):
    EPS = 1e-5   # EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    # max_iter = 30000
    iter = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        # if iter % 5000 == 0:
        #     print "Iter", iter
        #     print "t = ", t ,", f(x) = " ,(f(x, y, t))
        #     print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    print "Iter", iter
    print "t = ", t, ", f(x) = ", (f(x, y, t))
    print "Gradient: ", df(x, y, t), "\n"
    return t

def j(x, y, theta):
    return np.sum(np.sum((np.dot(theta.T, x) - y)**2, 0))

def dj(x, y, theta):
    return 2. * np.dot(x, (np.dot(theta.T, x) - y).T)

def test_functions(x, y, theta):

    results = []
    h = 1e-5
    for i in range(5):
        theta1 = theta.copy()
        theta2 = theta.copy()
        theta2[i, i] = theta1[i, i] + h

        result1 = (j(x, y, theta2) - j(x, y, theta1)) / (h)
        result2 = dj(x, y, theta1)
        results.append(abs(result1 == result2[i, i]) < 1e-2)
        # results.append((result1, result2[i, i]))

    return results

# move a sample of uncropped and cropped images for part 1
def part1():
    uncropped_files =["bracco3.jpg", "bracco4.jpg", "bracco33.jpg", "baldwin13.jpg"]
    cropped_files = ["carell23.jpeg", "hader12.jpg", "radcliffe77.jpg", "ferrera70.jpg", "harmon64.jpg", "drescher30.jpg"]

    if os.path.isdir("un" + actresses_folder):
        for i in uncropped_files[:3]:
            shutil.copyfile("un" + actresses_folder + i,figures_folder + i)

    if os.path.isdir("un" + actors_folder):
        shutil.copyfile("un" + actors_folder + uncropped_files[3],figures_folder + uncropped_files[3])

    for i in cropped_files:
        if os.path.exists(actresses_folder + i):
            shutil.copyfile(actresses_folder + i, figures_folder + i)
        elif os.path.exists(actors_folder + i):
            shutil.copyfile(actors_folder + i, figures_folder + i)


def part3_4():
    act = ["baldwin", "carell"]
    act_dict = get_images(actors_folder, act, 10, 10, 70)

    # create x, y, theta0 to use for f and df
    x = np.vstack((act_dict.get(act[0])[2], act_dict.get(act[1])[2]))
    x = x.T
    y = np.append(np.full((70,), 1), np.full((70, ), 0))
    theta0 = np.full((1025, ), 0.1)

    # run gradient descent
    theta1 = grad_descent(f, df, x, y, theta0, 0.000005)

    # cost function cost on training and validation sets
    x1 = np.vstack((act_dict.get(act[0])[1], act_dict.get(act[1])[1]))
    x1 = x1.T
    y1 = np.append(np.full((10,), 1), np.full((10,), 0))
    print "cost function cost on training set, validation set:", f(x, y, theta1), f(x1, y1, theta1)

    # test on training set and validation set
    results = []
    results.append([np.dot(theta1, i.T) > 0.5 for i in act_dict.get(act[0])[2]].count(True) * 100 / 70.)
    results.append([np.dot(theta1, i.T) > 0.5 for i in act_dict.get(act[0])[1]].count(True) * 100 / 10.)
    results.append([np.dot(theta1, i.T) < 0.5 for i in act_dict.get(act[1])[2]].count(True) * 100 / 70.)
    results.append([np.dot(theta1, i.T) < 0.5 for i in act_dict.get(act[1])[1]].count(True) * 100 / 10.)

    print results
    print "performance on training set, validation set:", (results[0] + results[2]) / 2, (results[1] + results[3]) / 2

    # part 4 a, use 2 images per actor, save theta as images
    x2 = np.vstack((act_dict.get(act[0])[2][0:2], act_dict.get(act[1])[2][0:2]))
    x2 = x2.T
    y2 = np.append(np.full((2,), 1), np.full((2, ), 0))
    theta2 = grad_descent(f, df, x2, y2, theta0, 0.000001)

    im = np.reshape(theta1[1:], (32, 32))
    im2 = np.reshape(theta2[1:], (32, 32))
    plt.imsave(figures_folder + "part4_a_dots_on_full_set" + ".jpg", im, cmap=plt.cm.coolwarm)
    plt.imsave(figures_folder + "part4_a_face_on_2" + ".jpg", im2, cmap=plt.cm.coolwarm)

    # part 4 b, theta image running on full set
    theta3 = grad_descent(f, df, x, y, theta0, 0.000005, 10)
    theta4 = grad_descent(f, df, x, y, theta0, 0.000005, 5000)
    im3 = np.reshape(theta3[1:], (32, 32))
    im4 = np.reshape(theta4[1:], (32, 32))
    plt.imsave(figures_folder + "part4_b_dots_on_full_set" + ".jpg", im4, cmap=plt.cm.coolwarm)
    plt.imsave(figures_folder + "part4_b_face_on_full_set" + ".jpg", im3, cmap=plt.cm.coolwarm)

def part5():
    act_names = ['baldwin', 'hader', 'carell', 'bracco', 'gilpin', 'harmon',
                 'butler', 'radcliffe', 'vartan', 'chenoweth', 'drescher', 'ferrera']

    # retrive data in act_dict
    act_dict = {}
    act_dict.update(get_images(actors_folder, act_names[:3], 10, 10, 250))
    act_dict.update(get_images(actresses_folder, act_names[3:6], 10, 10, 250))
    act_dict.update(get_images(actors_folder, act_names[6:9], 10, 50, 20))
    act_dict.update(get_images(actresses_folder, act_names[9:12], 10, 50, 20))

    # add training data to 3 lists
    act_training = []
    act_validation = []
    act2_validation = []
    for i in range(6):
        act_training.append(np.array(act_dict.get(act_names[i])[2]))
        act_validation.append(np.array(act_dict.get(act_names[i])[1]))
        act2_validation.append(np.array(act_dict.get(act_names[i + 6])[1]))

    max_training_size = 0
    max_training_size = min([len(i) for i in act_training])
    #max_training_size = 10

    theta0 = np.full((1025,), 0.3)
    results_dict = {}
    index = 1
    while index < max_training_size:

        # set data for current iteration with index image samples
        x = np.vstack([i[:index] for i in act_training])
        x = x.T
        y = np.append(np.full((index * 3,), 1), np.full((index * 3,), -1))
        theta1 = grad_descent(f, df, x, y, theta0, 0.000005)

        # check theta1 results
        training_results = []
        validation_results = []
        validation_results2 = []
        for i in range(3):
            training_results.append([np.dot(theta1.T, n) > 0. for n in
                            act_training[i][:index]].count(True) * 100. / (index))
            training_results.append([np.dot(theta1.T, n) < 0. for n in
                                     act_training[i + 3][:index]].count(True) * 100. / (index))

            validation_results.append([np.dot(theta1.T, n) > 0. for n in
                                       act_validation[i]].count(True) * 100. / (10))
            validation_results.append([np.dot(theta1.T, n) < 0. for n in
                                       act_validation[i + 3]].count(True) * 100. / (10))

            validation_results2.append([np.dot(theta1.T, n) > 0. for n in
                                        act2_validation[i]].count(True) * 100. / (50))
            validation_results2.append([np.dot(theta1.T, n) < 0. for n in
                                        act2_validation[i + 3]].count(True) * 100. / (50))

        results_dict[index] = [sum(training_results) / 6., sum(validation_results) / 6., sum(validation_results2) / 6.]
        index += 1

    lists = sorted(results_dict.items())
    y, x = zip(*lists)

    plt.figure(1)
    plt.plot(x, y)
    plt.xlabel("performance")
    plt.ylabel("sample size")
    plt.legend(["training set", "validation set n = 10", "other acts validation n = 50"])
    # plt.show()
    plt.savefig(figures_folder + "part5_plot.jpg")

    print results_dict

def part6_7_8():
    act_names = ['baldwin', 'hader', 'carell', 'bracco', 'gilpin', 'harmon']

    # retrive data in act_dict
    act_dict = {}
    act_dict.update(get_images(actors_folder, act_names[:3], 10, 10, 70))
    act_dict.update(get_images(actresses_folder, act_names[3:], 10, 10, 70))

    # add training data to act
    act = []
    for i in range(6):
        act.append(np.array(act_dict.get(act_names[i])[2]))

    # create x, y, theta0 to use for j and dj
    x = np.vstack(act)
    y = np.zeros((6, x.shape[0]))
    prev = 0
    for i in range(6):
        y[i, prev:prev + act[i].shape[0]] = 1.
        prev += act[i].shape[0]

    theta0 = np.ones((x.shape[1], 6)) * 0.001
    x = x.T

    # part 6d
    function_tests = test_functions(x, y, theta0)

    # part 7, run gradient descent
    theta1 = grad_descent(j, dj, x, y, theta0, 0.0000005)

    # test on training set and validation set

    training = []
    validation = []
    for i in range(6):
        name = act_names[i]
        training.append([np.dot(theta1.T, n).tolist().index(max(np.dot(theta1.T, n))) == i for n in
                       act_dict.get(name)[2]].count(True) * 100 / len(act_dict.get(name)[2]))
        validation.append([np.dot(theta1.T, n).tolist().index(max(np.dot(theta1.T, n))) == i for n in
                       act_dict.get(name)[1]].count(True) * 100 / len(act_dict.get(name)[1]))

    results = [training, validation]
    print results

    # get performance for trainings set, validation set
    print "perormance on training set, validation set:", (sum(results[0])/ 6.), (sum(results[1])/ 6.)

    # part 8 save theta images
    for i in range(6):
        im = np.reshape(theta1[1:, i], (32, 32))
        plt.imsave(figures_folder + "part8_" + act_names[i] + ".jpg", im, cmap=plt.cm.coolwarm)


# Main Function
if __name__ == '__main__':
    part1()
    part3_4()
    part5()
    part6_7_8()
