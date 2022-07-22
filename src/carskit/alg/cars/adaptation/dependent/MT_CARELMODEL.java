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
 * Multitask Context Aware Relation Based BPR
 *
 */

public class MT_CARELMODEL extends ContextRecommender {

    protected DenseVector condBias;

    protected static int numConditions;
    protected static ArrayList<Integer> EmptyContextConditions;

    public MT_CARELMODEL(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isCARSRecommender=true;
        numConditions = rateDao.numConditions();
        EmptyContextConditions = rateDao.getEmptyContextConditions();

        //isRankingPred = true;
        initByNorm = false;
        this.algoName = "MT_CARELMODEL";
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        condBias = new DenseVector(numConditions);
        condBias.init(initMean, initStd);
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception { //(same as in notes but with just qi, remove golbalMean, bu and bi)
        double predAux = DenseMatrix.rowMult(P, u, Q, j);
        for(int cond:getConditions(c)){
            predAux+=condBias.get(cond);
        }
        double pred = Math.exp(predAux) / (1 + Math.exp(predAux));
        return pred;
    }

    //@Override
    protected double predictRel(int u, int i, int j, int c) throws Exception { //new prediction from notes
        //double pred=globalMean + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j);
        //double pred = 0.0;

        double predAux = 0.0;
        int l = 0;
        for(int k = P.numColumns(); l < k; ++l) { //TODO l<k or numFactors??
            predAux += P.get(u, l) * (Q.get(i, l) - Q.get(j, l));

            if(predAux > 700){ //TODO P and some Q gets large which causes infinity
                System.out.println(P.get(u, l) + " ;;;; " + Q.get(i,l) + " ;;;; " + Q.get(j,l));
            }
        }

        double pre = predAux;

        for(int cond:getConditions(c)){
            predAux+=condBias.get(cond);
        }

        double pred = Math.exp(predAux) / (1 + Math.exp(predAux)); //TODO this gets to infinity due to P*(Qi-Qj)
        /*if(Double.isNaN(pred)){
            System.out.println("NOT A NUMBER");
            System.out.println(pre);
            System.out.println(predAux);
        }*/

        /*for (int f = 0; f < numFactors; f++) { //is this correct? How else can I get the i and j separately to substract one from the other?
            double puf = P.get(u, f);
            double qif = Q.get(i, f);
            double qjf = Q.get(j, f);

            pred += Math.exp(puf*(qif-qjf)) / (1 + Math.exp(puf*(qif-qjf)));
        }*/

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

                    double ruic = me1.get();
                    double rujc = me2.get();

                    double pi = 0.5;
                    if(ruic > rujc) pi = 1;
                    else if(ruic < rujc) pi = 0;

                    int i = rateDao.getItemIdFromUI(ui);
                    int j = rateDao.getItemIdFromUI(ui2);
                    if(i >= j) continue; //to make sure unique pairs are selected (1,2 ; 1,3 ; 2,3 ; but not 2,1 ; 3,2 ; 3,1

                    double piHat = predictRel(u1, i, j, ctx1);
                    System.out.println(piHat);

                    double euij = pi - piHat;

                    double pred1 = predict(u1, i, ctx1, false);
                    double eui = ruic - pred1;
                    double pred2 = predict(u2, j, ctx2, false);
                    double euj = rujc - pred2;

                    // Rating prediction
                    loss += alpha/2 * (eui * eui);
                    loss += alpha/2 * (euj * euj);

                    // Ranking
                    loss += (1-alpha) * (euij * euij);

                    /*double sgd = 0.0; //old formula
                    double bc_sum = 0;
                    for (int cond : getConditions(ctx1)) {
                        double bc = condBias.get(cond);
                        bc_sum += bc;
                        sgd = euij - regC * bc;
                        condBias.add(cond, lRate * sgd);
                    }
                    loss += regB * bc_sum;*/

                    double bc_sum = 0;
                    for (int cond : getConditions(ctx1)) {
                        double bc = condBias.get(cond);
                        bc_sum += bc;

                        //Only for rating
                        double sgd = euj - regC * bc;
                        condBias.add(cond, lRate * sgd);
                    }


                    for (int f = 0; f < numFactors; f++) {
                        double puf = P.get(u1, f);
                        double qif = Q.get(i, f);
                        double qjf = Q.get(j, f);

                        /*P.add(u1, f, lRate * ((qif-qjf)*(Math.exp(puf*(qif-qjf)))*((pi-1) * Math.exp(puf*(qif-qjf)) + pi)/Math.pow(Math.exp(puf*(qif-qjf)) + 1,3) + regU * puf));
                        Q.add(i, f, lRate * ((puf)*(Math.exp(puf*(qif-qjf)))*((pi-1) * Math.exp(puf*(qif-qjf)) + pi)/Math.pow(Math.exp(puf*(qif-qjf)) + 1,3) + regI * qif));
                        Q.add(j, f, lRate * ((puf)*(Math.exp(puf*(qif-qjf)))*((pi-1) * Math.exp(puf*(qif-qjf)) + pi)/Math.pow(Math.exp(puf*(qif-qjf)) + 1,3) + regI * qjf));*/

                        // Rating
                        P.add(u1, f, lRate * alpha * (euj * qjf - regU * puf));
                        Q.add(i, f, lRate * alpha * (eui * puf - regI * qif));
                        Q.add(j, f, lRate * alpha * (euj * puf - regI * qjf));

                        for (int cond : getConditions(ctx1)) {
                            double bc = condBias.get(cond);
                            /*P.add(u1, f, lRate * (alpha*qif*(globalMean + bu + bi + bc - ruic) + regU * puf)); //TODO what about the biases here?
                            Q.add(i, f, lRate * (alpha*puf*(globalMean + bu + bi + bc - ruic) + regI * qif));
                            P.add(u1, f, lRate * (alpha*qjf*(globalMean + bu + bj + bc - rujc) + regU * puf));
                            Q.add(j, f, lRate * (alpha*puf*(globalMean + bu + bj + bc - rujc) + regI * qjf));
                            */
                        }

                        // Ranking
                        double eAux = Math.exp(puf*(qif-qjf) + bc_sum);
                        double eDiv = eAux / (1+eAux);

                        P.add(u1, f, lRate * (alpha - 1) * ((((qif-qjf)*(pi-eDiv)*(pi-eDiv)*(eDiv))/(1+eAux)) - regU * puf));
                        Q.add(i, f, lRate * (alpha - 1) * ((((puf)*(pi-eDiv)*(pi-eDiv)*(eDiv))/(1+eAux)) - regI * qif));
                        Q.add(j, f, lRate * (alpha - 1) * ((((puf)*(pi-eDiv)*(pi-eDiv)*(eDiv))/(1+eAux)) - regI * qjf));

                        double sgd = 0.0;
                        for (int cond : getConditions(ctx1)) {
                            double bc = condBias.get(cond);
                            condBias.add(cond, lRate * (alpha - 1) * - (((eAux*((pi-1)*eAux + pi))/Math.pow(eAux + 1,3)) - regB * bc));
                        }

                        loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf + regB * bc_sum * bc_sum;
                    }
                }
            }

            loss *= 0.05;

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