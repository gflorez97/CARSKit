package carskit.alg.cars.adaptation.dependent;

import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;
import happy.coding.io.Strings;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;

import java.util.ArrayList;
import java.util.List;

/**
 * Multitask Context Aware BPR
 *
 */

public class MT_CABPR extends ContextRecommender {

    protected DenseVector condBias;
    protected DenseMatrix ucBias;
    protected DenseMatrix icBias;

    protected static int numConditions;
    protected static ArrayList<Integer> EmptyContextConditions;

    public MT_CABPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isCARSRecommender=true;
        numConditions = rateDao.numConditions();
        EmptyContextConditions = rateDao.getEmptyContextConditions();

        //isRankingPred = true;
        initByNorm = false;
        this.algoName = "MT_CABPR";
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        //userCache = train.rowCache(cacheSpec);

        userBias = new DenseVector(numUsers);
        userBias.init(initMean, initStd);

        itemBias = new DenseVector(numItems);
        itemBias.init(initMean, initStd);

        condBias = new DenseVector(numConditions);
        condBias.init(initMean, initStd);
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        double pred=globalMean + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j);

        for(int cond:getConditions(c)){
            pred+=condBias.get(cond);
        }

//        if(pred>100) { //TODO Big numbers causing infinity
//            System.out.println(DenseMatrix.rowMult(P, u, Q, j));
//            System.out.println(pred);
//            System.out.println("____________");
//        }
        return pred;
    }


    @Override
    protected void buildModel() throws Exception {

        double alpha = algoOptions.getFloat("-alpha");

        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;

            for (MatrixEntry me1 : trainMatrix) {
                for (MatrixEntry me2 : trainMatrix) {
                    int ctx1 = me1.column(); // context
                    int ctx2 = me2.column();
                    if(ctx1 != ctx2) continue;

                    int ui = me1.row(); // user-item
                    int ui2 = me2.row();
                    int u1 = rateDao.getUserIdFromUI(ui);
                    int u2 = rateDao.getUserIdFromUI(ui2);
                    if(u2 != u1) continue; // u2 has to be the same as u1, if not keep trying

                    int i = rateDao.getItemIdFromUI(ui);
                    int j = rateDao.getItemIdFromUI(ui2);
                    if(i >= j) continue; //I am trying to make sure combinations of items are not repeated ([0,1] and [1,0])

                    //System.out.println(u1 + " " + i + " " + j);

                    double ruic = me1.get();
                    double rujc = me2.get();
                    //if(ruic <= rujc) continue; //For the i and j item pairs, I should only be considering those in which rating of i is greater than rating of j.

                    if(ruic <= rujc){ //invert the items, as per above only one side of the combination will be considered
                        double aux = ruic;
                        ruic = rujc;
                        rujc = aux;
                        int aux2 = i;
                        i = j;
                        j = aux2;
                    }

                    double pred1 = predict(u1, i, ctx1, false);
                    double eui = ruic - pred1;



                    double pred2 = predict(u2, j, ctx2, false);
                    double euj = rujc - pred2;
                    double xuij = pred1 - pred2;

                    // Rating prediction
                    loss += alpha/2 * (eui * eui);
                    loss += alpha/2 * (euj * euj);

                    // Ranking
                    double vals = -Math.log(g(xuij));
                    loss += (1-alpha) * vals;

                    double bu = userBias.get(u1); //bu1 == bu2
                    double sgd = eui - regB * bu;
                    double sgd2 = euj - regB * bu;
                    userBias.add(u1, lRate * sgd);
                    loss += regB * bu * bu;

                    double bi = itemBias.get(i);
                    sgd = eui - regB * bi;
                    itemBias.add(i, lRate * sgd);
                    double bj = itemBias.get(j);
                    sgd2 = euj - regB * bj;
                    itemBias.add(j, lRate * sgd2);
                    loss += regB * bi * bi + regB * bj * bj;

                    double bc_sum = 0;
                    for (int cond : getConditions(ctx1)) {
                        double bc = condBias.get(cond);
                        bc_sum += bc;
                        sgd = eui - regC * bc;
                        sgd2 = euj - regC * bc;
                        condBias.add(cond, lRate * sgd);
                        condBias.add(cond, lRate * sgd2);
                    }
                    loss += regB * bc_sum;

                    double cmg = g(-xuij);
                    for (int f = 0; f < numFactors; f++) {
                        double puf = P.get(u1, f);
                        double qif = Q.get(i, f);
                        double qjf = Q.get(j, f);

                        // Rating
                        P.add(u1, f, lRate * alpha * (euj * qjf - regU * puf) ); //TODO + (alpha-1) ...);
                        Q.add(i, f, lRate * alpha * (eui * puf - regI * qif));
                        Q.add(j, f, lRate * alpha * (euj * puf - regI * qjf));

                        //TODO the one above or this one?
//                        for (int cond : getConditions(ctx1)) {
//                            double bc = condBias.get(cond);
//                            P.add(u1, f, lRate * (alpha*qif*(globalMean + bu + bi + bc - ruic) - regU * puf));
//                            Q.add(i, f, lRate * (alpha*puf*(globalMean + bu + bi + bc - ruic) - regI * qif));
//                            P.add(u1, f, lRate * (alpha*qjf*(globalMean + bu + bj + bc - rujc) - regU * puf));
//                            Q.add(j, f, lRate * (alpha*puf*(globalMean + bu + bj + bc - rujc) - regI * qjf));
//                        }

                        // Ranking
                        P.add(u1, f, lRate * (alpha - 1) * (Math.exp(pred2) * (qif-qjf)/(Math.exp(pred1)+Math.exp(pred2)) - regU * puf));
                        Q.add(i, f, lRate * (alpha - 1) * (Math.exp(pred2) * (puf)/(Math.exp(pred1)+Math.exp(pred2)) - regI * qif));
                        Q.add(j, f, lRate * (alpha - 1) * (Math.exp(pred1) * (puf)/(Math.exp(pred1)+Math.exp(pred2)) - regI * qjf));


                        loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
                    }
                }
            }

            loss*=0.05;

            if (isConverged(iter))
                break;

        }
    }

    protected List<Integer> getConditions(int ctx)
    {
        String context=rateDao.getContextId(ctx);
        String[] cts = context.split(",");
        List<Integer> conds = new ArrayList<>();
        for(String ct:cts)
            conds.add(Integer.valueOf(ct));
        return conds;
    }

    @Override
    public String toString() {
        return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, numIters }, ",");
    }
}
