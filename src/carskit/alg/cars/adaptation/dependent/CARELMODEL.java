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
 * Context Aware Relation Based BPR
 *
 */

public class CARELMODEL extends ContextRecommender {

    protected DenseVector condBias;
    protected DenseMatrix ucBias;
    protected DenseMatrix icBias;

    protected static int numConditions;
    protected static ArrayList<Integer> EmptyContextConditions;

    public CARELMODEL(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isCARSRecommender=true;
        numConditions = rateDao.numConditions();
        EmptyContextConditions = rateDao.getEmptyContextConditions();

        isRankingPred = true;
        initByNorm = false;
        this.algoName = "CARELBPR";
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
    protected double predictRel(int u, int i, int j, int c) throws Exception {
        //double pred=globalMean + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j);
        double pred = 0.0;

        //puRow = P.row(u);
        //qiRow = Q.row(i);


        for (int f = 0; f < numFactors; f++) { //TODO is this correct? How else can i get the i and j sepparately to substract one from the other
            double puf = P.get(u, f);
            double qif = Q.get(i, f);
            double qjf = Q.get(j, f);

            pred += Math.exp(puf*(qif-qjf)) / (1 + Math.exp(puf*(qif-qjf))); //TODO problem is here

        }
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

                    double pi = 0.5;
                    if(ruic > rujc) pi = 1;
                    else if(ruic < rujc) pi = 0;

                    int i = rateDao.getItemIdFromUI(ui);
                    int j = rateDao.getItemIdFromUI(ui2);
                    double piHat = predictRel(u1, i, j, ctx1);

                    double euij = Math.pow(pi - piHat, 2);

                    loss += euij/2;

                    double bu = userBias.get(u1); //bu1 == bu2
                    double sgd = euij - regB * bu;
                    double bc_sum = 0;
                    for (int cond : getConditions(ctx1)) {
                        double bc = condBias.get(cond);
                        bc_sum += bc;
                        sgd = euij - regC * bc;
                        condBias.add(cond, lRate * sgd);
                    }

                    for (int f = 0; f < numFactors; f++) {
                        double puf = P.get(u1, f);
                        double qif = Q.get(i, f);
                        double qjf = Q.get(j, f);

                        P.add(u1, f, lRate * ((qif-qjf)*(Math.exp(puf*(qif-qjf)))*((pi-1) * Math.exp(puf*(qif-qjf)) + pi))/Math.pow(Math.exp(puf*(qif-qjf)) + 1,3));
                        Q.add(i, f, lRate * ((puf)*(Math.exp(puf*(qif-qjf)))*((pi-1) * Math.exp(puf*(qif-qjf)) + pi))/Math.pow(Math.exp(puf*(qif-qjf)) + 1,3));
                        Q.add(j, f, lRate * ((puf)*(Math.exp(puf*(qif-qjf)))*((pi-1) * Math.exp(puf*(qif-qjf)) + pi))/Math.pow(Math.exp(puf*(qif-qjf)) + 1,3));

                        //TODO should I also add matrices to store and update bu and Bc ??

                        loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
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