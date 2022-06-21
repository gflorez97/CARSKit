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
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;

import carskit.data.structure.SparseMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Pair Scores BPR
 *
 */

public class CAPAIRSMF extends ContextRecommender {

    protected DenseVector condBias;
    protected DenseMatrix ucBias;
    protected DenseMatrix icBias;

    protected static int numConditions;
    protected static ArrayList<Integer> EmptyContextConditions;

    public CAPAIRSMF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isCARSRecommender=true;
        numConditions = rateDao.numConditions();
        EmptyContextConditions = rateDao.getEmptyContextConditions();

        isRankingPred = true;
        initByNorm = false;
        this.algoName = "CAPAIRSMF";
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

    //@Override
    protected double predictPair(int u, int i, int j, int c) throws Exception {
        double pred = 0;//double pred=globalMean + userBias.get(u) + itemBias.get(i,j) + DenseMatrix.rowMult(P, u, Q, i,j); //TODO

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

                    double ruic = me1.get();
                    double rujc = me2.get();
                    double ruijc = ruic - rujc;

                    int i = rateDao.getItemIdFromUI(ui);
                    int j = rateDao.getItemIdFromUI(ui2);

                    double pred = predictPair(u1, i, j, ctx1); //TODO adapt the method to the pair score situation
                    double euij = ruijc - pred;
                    double xuij = euij;

                    double vals = -Math.log(g(xuij));
                    loss += vals;

                    double bu = userBias.get(u1); //bu1 == bu2
                    double sgd = euij - regB * bu;
                    double bc_sum = 0;
                    for (int cond : getConditions(ctx1)) {
                        double bc = condBias.get(cond);
                        bc_sum += bc;
                        sgd = euij - regC * bc;
                        condBias.add(cond, lRate * sgd);
                    }

                    double cmg = g(-xuij);

                    for (int f = 0; f < numFactors; f++) {
                        double puf = P.get(u1, f);
                        double qijf = 0.0; //double qijf = Q.get(i,j, f); //TODO

                        P.add(u1, f, lRate * ((-qijf) - regI * qijf));
                        Q.add(j, f, lRate * ((-puf) - regI * qijf));

                        //TODO should I also add matrices to store and update bu and Bc ??

                        loss += regU * puf * puf + regI * qijf * qijf;
                    }

                }
            }

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