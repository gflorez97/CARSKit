package carskit.alg.cars.adaptation.dependent;

// Copyright (C) 2015 Yong Zheng
//
// This file is part of CARSKit.
//
// CARSKit is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CARSKit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CARSKit. If not, see <http://www.gnu.org/licenses/>.
//

import carskit.generic.ContextRecommender;
import happy.coding.io.Strings;
import happy.coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseVector;

import carskit.data.structure.SparseMatrix;
import carskit.generic.IterativeRecommender;

import java.util.*;

/**
 * Context Aware BPR
 *
 */

public class CABPR extends ContextRecommender {

    protected DenseVector condBias;
    protected DenseMatrix ucBias;
    protected DenseMatrix icBias;

    protected static int numConditions;
    protected static ArrayList<Integer> EmptyContextConditions;

    public CABPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isCARSRecommender=true;
        numConditions = rateDao.numConditions();
        EmptyContextConditions = rateDao.getEmptyContextConditions();

        //isRankingPred = true;
        initByNorm = false;
        this.algoName = "CABPR";
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
    protected double predict(int u, int j, int c) throws Exception { //TODO getting better results with the original basic predict function
        double pred=globalMean + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j);

        for(int cond:getConditions(c)){
            pred+=condBias.get(cond);
        }
        return pred;
    }


    @Override
    protected void buildModel() throws Exception {

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

                    double vals = -Math.log(g(xuij));
                    loss += vals;

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

                        //TODO original BPR above, with my derivatives (maybe wrong?) below
                        /*P.add(u1, f, lRate * (cmg * (qif - qjf) - regU * puf));
                        Q.add(i, f, lRate * (cmg * puf - regI * qif));
                        Q.add(j, f, lRate * (cmg * (-puf) - regI * qjf));*/

                        /*P.add(u1, f, lRate * (Math.exp(puf*qjf) * (qif-qjf)/(Math.exp(puf*qif)+Math.exp(puf*qjf)) + regU * puf));
                        Q.add(i, f, lRate * (Math.exp(puf*qjf) * (puf)/(Math.exp(puf*qif)+Math.exp(puf*qjf)) + regI * qif));
                        Q.add(j, f, lRate * (Math.exp(puf*qif) * (puf)/(Math.exp(puf*qif)+Math.exp(puf*qjf)) + regI * qjf));*/

                        P.add(u1, f, lRate * (Math.exp(pred2) * (qif-qjf)/(Math.exp(pred1)+Math.exp(pred2)) + regU * puf));
                        Q.add(i, f, lRate * (Math.exp(pred2) * (puf)/(Math.exp(pred1)+Math.exp(pred2)) + regI * qif));
                        Q.add(j, f, lRate * (Math.exp(pred1) * (puf)/(Math.exp(pred1)+Math.exp(pred2)) + regI * qjf));


                        loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
                    }
                }
            }

            loss*=0.5;

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