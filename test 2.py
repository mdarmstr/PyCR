    class_variables = scale_training_sample[:, valid_idx]
    dummyU, dummyS, V = svds(class_variables, k=2)
    scale_all_sample = scale_all_sample.transpose()
    score = np.dot(scale_all_sample,V)

    for z in range(1, classNum+1):
        class_score = score[class_index_list[z],:]
        x_ellipse, y_ellipse = confident_ellipse(class_score[:, 0], class_score[:, 1])
        plt.plot(x_ellipse, y_ellipse,color=class_color[z-1])
        plt.fill(x_ellipse, y_ellipse,color=class_color[z-1], alpha=0.3)
        class_Xt = score[class_index_list[z], :]
        plt.scatter(class_Xt[:, 0], class_Xt[:, 1], c=class_color[z-1], marker=class_label[0], label='training' + str(z))
    # calculating the PCA percentage value
    pU, pS, pV = np.linalg.svd(class_variables)
    pca_percentage_val = np.cumsum(pS) / sum(pS)
    p1_percentage = pca_percentage_val[0] * 100
    p2_percentage = pca_percentage_val[1] * 100
    plt.xlabel("P1 \n P1: = {0:0.3f}".format(p1_percentage)+"%")
    plt.ylabel("P2 \n P2: = {0:0.3f}".format(p2_percentage)+"%")
    plt.rcParams.update({'font.size': 21})
    plt.title('PCA_training')
    plt.legend()
    plt.savefig('output/pca_taining.png')
    if isexternal:
        external_Xt = np.dot(scaled_external[:,valid_idx], V)
        for n in range(1, classNum+1):
            class_external_Xt = external_Xt[external_class_index_list[n], :]
            plt.scatter(class_external_Xt[:, 0], class_external_Xt[:, 1], c=class_color[n-1], marker=class_label[1],
                               label='external' + str(n))
        clf_extern = svm.SVC(kernel='linear', random_state=0, probability=True)
        clf_extern.fit(sampleList[:,valid_idx], classList)
        class_pred = clf_extern.predict(external_validation[:, valid_idx])
        classofic_report = classification_report(external_class, class_pred)
        plt.title('PCA_Training_VS_Validation')
        plt.rcParams.update({'font.size': 21})
        plt.legend()
        plt.savefig('output/pca_external.png')
        plt.figure().clear()