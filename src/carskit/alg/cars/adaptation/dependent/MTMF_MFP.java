package carskit.alg.cars.adaptation.dependent;

import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SymmMatrix;

/**
 * Multitask Matrix Factorization - Pair Scores
 *
 * Kalloori, S., Ricci, F., & Tkalcic, M. (2016, September). Pairwise preferences based matrix factorization and nearest neighbor recommendation techniques. In Proceedings of the 10th ACM Conference on Recommender Systems (pp. 143-146).
 *
 * @author Gonzalo Florez Arias
 *
 */

public class MTMF_MFP extends ContextRecommender {

    // members for deviation-based models
    protected DenseVector condBias;
    protected DenseMatrix ucBias;
    protected DenseMatrix icBias;

    // members for similarity-based models
    protected SymmMatrix ccMatrix_ICS;
    protected DenseMatrix cfMatrix_LCS;
    protected DenseVector cVector_MCS;

    // factor for rating vs ranking
    double alpha;

    public MTMF_MFP(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
    }

    protected void initModel() throws Exception {

        super.initModel();

        userBias = new DenseVector(numUsers);
        userBias.init(initMean, initStd);

        itemBias = new DenseVector(numItems);
        itemBias.init(initMean, initStd);

        itemBiasPairs = new DenseVector(numItems*numItems); //TODO how to store pairs in vector, size probably needs to be constrained
        itemBiasPairs.init(initMean, initStd);

        condBias = new DenseVector(numConditions);
        condBias.init(initMean, initStd);

    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        double pred=globalMean + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j);
        for(int cond:getConditions(c)){
            pred+=condBias.get(cond);
        }
        return pred;
    }

    protected double predictPair(int u, int ij, int c) throws Exception { //TODO
        double pred=globalMean + userBias.get(u) + itemBiasPairs.get(ij) + DenseMatrix.rowMult(P, u, Qp, ij);
        for(int cond:getConditions(c)){
            pred+=condBias.get(cond);
        }
        return pred;
    }

    @Override
    protected void buildModel() throws Exception {

        alpha = algoOptions.getFloat("-alpha");

        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;
            for (MatrixEntry me1 : trainMatrix) {
                for (MatrixEntry me2 : trainMatrix) {

                    int ui = me1.row(); // user-item
                    int ui2 = me2.row();
                    int u1 = rateDao.getUserIdFromUI(ui);
                    int u2 = rateDao.getUserIdFromUI(ui2);
                    if(u2 != u1) continue; // u2 has to be the same as u1, if not keep trying

                    double rujc1 = me1.get();
                    double rujc2 = me2.get();
                    if((rujc1 > 2 && rujc2 > 2) || (rujc1 <= 2 && rujc2 <= 2)) continue; // Only a positive and a negative item can be paired


                    // RATING PREDICTION
                    int j1 = rateDao.getItemIdFromUI(ui);
                    int ctx1 = me1.column(); // context

                    double pred1 = predict(u1, j1, ctx1, false);
                    double euj1 = rujc1 - pred1;

                    loss += euj1 * euj1 * alpha;

                    // update factors
                    double bu = userBias.get(u1);
                    double sgd = euj1 - regB * bu;
                    userBias.add(u1, lRate * sgd);

                    loss += regB * bu * bu * alpha;


                    double bj = itemBias.get(j1);
                    sgd = euj1 - regB * bj;
                    itemBias.add(j1, lRate * sgd);

                    loss += regB * bj * bj * alpha;

                    double bc_sum = 0;
                    for (int cond : getConditions(ctx1)) {
                        double bc = condBias.get(cond);
                        bc_sum += bc;
                        sgd = euj1 - regC * bc;
                        condBias.add(cond, lRate * sgd);
                    }

                    loss += regB * bc_sum * alpha;

                    for (int f = 0; f < numFactors; f++) {
                        double puf = P.get(u1, f);
                        double qjf = Q.get(j1, f);

                        double delta_u = euj1 * qjf - regU * puf;
                        double delta_j = euj1 * puf - regI * qjf;

                        P.add(u1, f, lRate * delta_u);
                        Q.add(j1, f, lRate * delta_j);

                        loss += (regU * puf * puf + regI * qjf * qjf) * alpha;
                    }

                    // PAIRWISE RANKING (MFP)
                    int j2 = rateDao.getItemIdFromUI(ui2);
                    int ctx2 = me2.column(); // context

                    double rPred = predictPair(u1,j1+j2,ctx1); //TODO

                    /*double pred2 = predict(u2, j2, ctx2, false);
                    double xuij = pred1 - pred2;

                    double vals = -Math.log(g(xuij));
                    loss += vals * (1-alpha);

                    double cmg = g(-xuij);
                    for (int f = 0; f < numFactors; f++) {
                        double puf = P.get(u1, f);
                        double qif = Q.get(j1, f);
                        double qjf = Q.get(j2, f);

                        P.add(u1, f, lRate * (cmg * (qif - qjf) - regU * puf));
                        Q.add(j1, f, lRate * (cmg * puf - regI * qif));
                        Q.add(j2, f, lRate * (cmg * (-puf) - regI * qjf));

                        loss += (regU * puf * puf + regI * qif * qif + regI * qjf * qjf) * (1-alpha);
                    }*/

                }
                loss *= 0.5;

                if (isConverged(iter))
                    break;
            }
        }// end of training

    }

}
