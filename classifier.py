if __name__ == '__main__':
    from sklearn import datasets
    # load from internet data to train and predict about dataset: https://archive.ics.uci.edu/ml/datasets/wine
    X_train = datasets.load_wine()
    print X_train["DESCR"]
    from sklearn.model_selection import train_test_split

    # split data set to "train" part and part "to predict" also shuffle data set.
    X_train, X_test, y_train, y_test = train_test_split(X_train["data"],
                                                        X_train["target"],
                                                        test_size=0.5)
    # import classifiers
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB

    # define classifiers
    forest = RandomForestClassifier()
    knn = KNeighborsClassifier()
    process = GaussianProcessClassifier()
    tree = DecisionTreeClassifier()
    bayes = GaussianNB()
    # train our classifiers
    pattern = "Finished training %s, start train: %s"
    print "Start train Forest."
    forest.fit(X_train, y_train)
    print(pattern % ("Forest", "KNN"))
    knn.fit(X_train, y_train)
    print(pattern % ("KNN", "Process"))
    process.fit(X_train, y_train)
    print(pattern % ("Process", "Tree"))
    tree.fit(X_train, y_train)
    print(pattern % ("Tree", "Bayes"))
    bayes.fit(X_train, y_train)
    print("Finish training Bayes.")
    # make a predictions
    data = X_test
    target = y_test
    print("All classifiers trained. Start making predictions")
    pattern = "Making predictions for %s"
    print(pattern % "Forest")
    forest_results = forest.predict(X_test)
    print(pattern % "KNN")
    knn_results = knn.predict(X_test)
    print(pattern % "Process")
    process_results = process.predict(X_test)
    print(pattern % "Tree")
    tree_results = tree.predict(X_test)
    print(pattern % "Bayes")
    bayes_results = bayes.predict(X_test)
    # print metrics
    from sklearn.metrics import classification_report
    print "Forest results: "
    print classification_report(y_true=y_test, y_pred=forest_results)
    print "KNN results: "
    print classification_report(y_true=y_test, y_pred=knn_results)
    print "Gaussian process results: "
    print classification_report(y_true=y_test, y_pred=process_results)
    print "Decision Tree results: "
    print classification_report(y_true=y_test, y_pred=tree_results)
    print "Bayes results: "
    print classification_report(y_true=y_test, y_pred=bayes_results)
