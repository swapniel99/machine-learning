function [l_mm_acc, l_mm_c] = cross_validate(X, y, C_range, k, trainer)
    m = size(X,1);
    c = cvpartition(m,'Kfold',k);
    Cl = length(C_range);
    accs = zeros(1,Cl);
    for ci = 1:Cl
        kacc = zeros(1,k);
        for i = 1:k
            X_tr = X(c.training(i),:);
            y_tr = y(c.training(i),:);

            X_te = X(c.test(i),:);
            y_te = y(c.test(i),:);

            [W, b] = trainer(X_tr,y_tr,C_range(ci));

            res = y_te.*sign(X_te*W-b);

            kacc(i) = sum(res==1)/length(res);
        end
        accs(ci) = mean(kacc);
    end
    [l_mm_acc, mi] = max(accs);
    l_mm_c = C_range(mi);
end
