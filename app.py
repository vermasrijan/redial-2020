######################################################################################
# Author: Srijan Verma, BITS Pilani, India                                           #
# Code developed in Sirimulla Research Group (http://sirimullaresearchgroup.com/)    #
# University of Texas at El Paso, Tx, USA                                            #
# Last modified: 25/08/2020                                                          #
######################################################################################

from flask import Flask, jsonify, render_template, request, url_for, redirect, make_response
import cairosvg
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import base64
import io
from PIL import Image
from run_script import Predict, Similarity, preprocess_smi, OchemToolALOGPS, FetchPhysicoProperty, FetchChemoDB, CheckInput, OchemAPIResults
import sys
from rdkit.Chem import rdDepictor
import json
import pickle
from  collections import OrderedDict

USE_OCHEM_API = True # If True, ochem API will be used for ALOGPS calculations (instead of ochem Tool)

app = Flask(__name__)


@app.route("/")
def home():

    return jsonify({'message': 'SERVER IS RUNNING'})

@app.route("/predict", methods=['GET', 'POST'])
def predict():

    if request.method in ['GET', 'POST']:
        # request = add_header(request)
        _input = request.form['smiles']
        
        print("PREDICTING FOR {}".format(_input), file=sys.stderr)
        
        # smiles = 'CCCCCC'
        all_dict = {}
        
        if _input == '' or _input == None or len(_input) == 0:
            all_dict['error'] = {'message': "INPUT ERROR", 'error': "EMPTY INPUT"}
            all_dict['error_flag'] = 'TRUE'
            print("ERROR: EMPTY INPUT")
            return jsonify(all_dict)#jsonify({'message': "SMILES ERROR", 'error': "EMPTY SMILES"})

        #Check input type
        smiles, synonyms, smi_flag, drug_name_flag, pubchem_cid_flag = CheckInput().check_input(_input)

        # Preprocessing the smiles
        processed_smiles = preprocess_smi(smiles)
        
        if processed_smiles == None:
            all_dict['error'] = {'message': "INPUT ERROR", 'error': "INVALID INPUT"}
            all_dict['error_flag'] = 'TRUE'
            print("ERROR: INVALID INPUT")
            return jsonify(all_dict)#jsonify({'message': "SMILES ERROR", 'error': "EMPTY SMILES"})

        try:
            # Below, getting attributes for using processed smiles -->
            p = Predict()
            s = Similarity()
            similarity_dict = s.model_initialization(processed_smiles, three_cl=False)
            three_cl_similarity_dict = s.model_initialization(processed_smiles, three_cl=True)
            predict_dict = p.model_initialization([processed_smiles])
            molecular_wt = FetchPhysicoProperty().get_molecular_wt(processed_smiles)
            molecular_formula = FetchPhysicoProperty().get_molecular_formula(processed_smiles)

            # Below, getting attributes for using query smiles -->
            pubchem_link, pubchem_cid = FetchChemoDB().fetch_pubchem(smiles)
            drug_central_link, drug_central_id = FetchChemoDB().fetch_drug_central(smiles, _input)

            # Debug 1
            # print('PRE EDITING BELOW =>')
            # print('Similairty BELOW =>')
            # pprint.pprint(similarity_dict)
            #
            # print('\nPrediction BELOW =>')
            # pprint.pprint(predict_dict)
            #
            # print('\nALL_DICT BELOW =>')
            # pprint.pprint(all_dict)

            ############<CONSENSUS MODEL CODE BELOW>########################
            consensus_prediction_results = {}
            consensus_prediction_results[processed_smiles] = {}
            act_labels = []
            act_proba_1 = []
            act_proba_0 = []
            tox_labels = []
            tox_proba_1 = []
            tox_proba_0 = []
            truhit_labels = []
            truhit_proba_1 = []
            truhit_proba_0 = []
            alpha_lisa_and_truhit_labels = []
            alpha_lisa_and_truhit_proba_1 = []
            alpha_lisa_and_truhit_proba_0 = []
            alpha_lisa_labels = []
            alpha_lisa_proba_1 = []
            alpha_lisa_proba_0 = []
            ace2_enzymatic_labels = []
            ace2_enzymatic_proba_1 = []
            ace2_enzymatic_proba_0 = []
            three_cl_enzymatic_labels = []
            three_cl_enzymatic_proba_1 = []
            three_cl_enzymatic_proba_0 = []

            for key, val in predict_dict.items():
                for key2, val2 in val.items():
                    if val[key2]['prediction'] == 'active' and 'Act' in key2:
                        proba_1 = float(val[key2]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        act_proba_1.append(proba_1)
                        act_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'ACTIVE'})
                        act_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'inactive' and 'Act' in key2:
                        proba_0 = float(val[key2]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        act_proba_1.append(proba_1)
                        act_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'INACTIVE'})
                        act_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'active' and 'Tox' in key2:
                        proba_1 = float(val[key2]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        tox_proba_1.append(proba_1)
                        tox_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'ACTIVE'})
                        tox_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'inactive' and 'Tox' in key2:
                        proba_0 = float(val[key2]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        tox_proba_1.append(proba_1)
                        tox_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'INACTIVE'})
                        tox_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'active' and (key2 == 'TruHitDes' or key2 == 'TruHitTopo' or key2 == 'TruHitFP'):
                        proba_1 = float(val[key2]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        truhit_proba_1.append(proba_1)
                        truhit_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'ACTIVE'})
                        truhit_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'inactive' and (key2 == 'TruHitDes' or key2 == 'TruHitTopo' or key2 == 'TruHitFP'):
                        proba_0 = float(val[key2]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        truhit_proba_1.append(proba_1)
                        truhit_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'INACTIVE'})
                        truhit_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'active' and (key2 == 'AlphaLisaAndTruHitDes' or key2 == 'AlphaLisaAndTruHitTopo' or key2 == 'AlphaLisaAndTruHitFP'):
                        proba_1 = float(val[key2]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        alpha_lisa_and_truhit_proba_1.append(proba_1)
                        alpha_lisa_and_truhit_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'ACTIVE'})
                        alpha_lisa_and_truhit_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'inactive' and (key2 == 'AlphaLisaAndTruHitDes' or key2 == 'AlphaLisaAndTruHitTopo' or key2 == 'AlphaLisaAndTruHitFP'):
                        proba_0 = float(val[key2]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        alpha_lisa_and_truhit_proba_1.append(proba_1)
                        alpha_lisa_and_truhit_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'INACTIVE'})
                        alpha_lisa_and_truhit_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'active' and (key2 == 'AlphaLisaDes' or key2 == 'AlphaLisaTopo' or key2 == 'AlphaLisaFP'):
                        proba_1 = float(val[key2]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        alpha_lisa_proba_1.append(proba_1)
                        alpha_lisa_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'ACTIVE'})
                        alpha_lisa_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'inactive' and (key2 == 'AlphaLisaDes' or key2 == 'AlphaLisaTopo' or key2 == 'AlphaLisaFP'):
                        proba_0 = float(val[key2]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        alpha_lisa_proba_1.append(proba_1)
                        alpha_lisa_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'INACTIVE'})
                        alpha_lisa_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'active' and (key2 == 'ACE2EnzymaticDes' or key2 == 'ACE2EnzymaticTopo' or key2 == 'ACE2EnzymaticFP'):
                        proba_1 = float(val[key2]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        ace2_enzymatic_proba_1.append(proba_1)
                        ace2_enzymatic_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'ACTIVE'})
                        ace2_enzymatic_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'inactive' and (key2 == 'ACE2EnzymaticDes' or key2 == 'ACE2EnzymaticTopo' or key2 == 'ACE2EnzymaticFP'):
                        proba_0 = float(val[key2]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        ace2_enzymatic_proba_1.append(proba_1)
                        ace2_enzymatic_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'INACTIVE'})
                        ace2_enzymatic_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'active' and (key2 == '3CLEnzymaticDes' or key2 == '3CLEnzymaticTopo' or key2 == '3CLEnzymaticFP'):
                        proba_1 = float(val[key2]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        three_cl_enzymatic_proba_1.append(proba_1)
                        three_cl_enzymatic_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'ACTIVE'})
                        three_cl_enzymatic_labels.append(val[key2]['prediction'])
                        continue

                    elif val[key2]['prediction'] == 'inactive' and (key2 == '3CLEnzymaticDes' or key2 == '3CLEnzymaticTopo' or key2 == '3CLEnzymaticFP'):
                        proba_0 = float(val[key2]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        three_cl_enzymatic_proba_1.append(proba_1)
                        three_cl_enzymatic_proba_0.append(proba_0)
                        predict_dict[key][key2].update({'predict_proba_0': str(proba_0)})
                        predict_dict[key][key2].update({'predict_proba_1': str(proba_1)})
                        predict_dict[key][key2].update({'prediction': 'INACTIVE'})
                        three_cl_enzymatic_labels.append(val[key2]['prediction'])
                        continue

            consensus_prediction_results[processed_smiles]['Activity Model'] = {'prediction': max(act_labels, key=act_labels.count)}
            consensus_prediction_results[processed_smiles]['Toxicity Model'] = {'prediction': max(tox_labels, key=tox_labels.count)}
            consensus_prediction_results[processed_smiles]['TruHit Model'] = {'prediction': max(truhit_labels, key=truhit_labels.count)}
            consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model'] = {'prediction': max(alpha_lisa_and_truhit_labels, key=alpha_lisa_and_truhit_labels.count)}
            consensus_prediction_results[processed_smiles]['AlphaLisa Model'] = {'prediction': max(alpha_lisa_labels, key=alpha_lisa_labels.count)}
            consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model'] = {'prediction': max(ace2_enzymatic_labels, key=ace2_enzymatic_labels.count)}
            consensus_prediction_results[processed_smiles]['3CL Enzymatic Model'] = {'prediction': max(three_cl_enzymatic_labels, key=three_cl_enzymatic_labels.count)}

            if consensus_prediction_results[processed_smiles]['Toxicity Model']['prediction'] == 'INACTIVE':
                consensus_prediction_results[processed_smiles]['Toxicity Model'].update({'probability': str(round(sum(tox_proba_0) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['Toxicity Model']['prediction'] == 'ACTIVE':
                consensus_prediction_results[processed_smiles]['Toxicity Model'].update({'probability': str(round(sum(tox_proba_1) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['Activity Model']['prediction'] == 'INACTIVE':
                consensus_prediction_results[processed_smiles]['Activity Model'].update({'probability': str(round(sum(act_proba_0) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['Activity Model']['prediction'] == 'ACTIVE':
                consensus_prediction_results[processed_smiles]['Activity Model'].update({'probability': str(round(sum(act_proba_1) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['TruHit Model']['prediction'] == 'INACTIVE':
                consensus_prediction_results[processed_smiles]['TruHit Model'].update({'probability': str(round(sum(truhit_proba_0) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['TruHit Model']['prediction'] == 'ACTIVE':
                consensus_prediction_results[processed_smiles]['TruHit Model'].update({'probability': str(round(sum(truhit_proba_1) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model']['prediction'] == 'INACTIVE':
                consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model'].update({'probability': str(round(sum(alpha_lisa_and_truhit_proba_0) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model']['prediction'] == 'ACTIVE':
                consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model'].update({'probability': str(round(sum(alpha_lisa_and_truhit_proba_1) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['AlphaLisa Model']['prediction'] == 'INACTIVE':
                consensus_prediction_results[processed_smiles]['AlphaLisa Model'].update({'probability': str(round(sum(alpha_lisa_proba_0) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['AlphaLisa Model']['prediction'] == 'ACTIVE':
                consensus_prediction_results[processed_smiles]['AlphaLisa Model'].update({'probability': str(round(sum(alpha_lisa_proba_1) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model']['prediction'] == 'INACTIVE':
                consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model'].update({'probability': str(round(sum(ace2_enzymatic_proba_0) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model']['prediction'] == 'ACTIVE':
                consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model'].update({'probability': str(round(sum(ace2_enzymatic_proba_1) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['3CL Enzymatic Model']['prediction'] == 'INACTIVE':
                consensus_prediction_results[processed_smiles]['3CL Enzymatic Model'].update({'probability': str(round(sum(three_cl_enzymatic_proba_0) / 3, 2))})

            if consensus_prediction_results[processed_smiles]['3CL Enzymatic Model']['prediction'] == 'ACTIVE':
                consensus_prediction_results[processed_smiles]['3CL Enzymatic Model'].update({'probability': str(round(sum(three_cl_enzymatic_proba_1) / 3, 2))})

            # Selecting the best model by looking at val/test results ; That is, below models DO NOT have consensus predictions!!!
            consensus_prediction_results[processed_smiles]['AlphaLisa Model'] = predict_dict[processed_smiles]['AlphaLisaDes']
            consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model'] = predict_dict[processed_smiles]['ACE2EnzymaticDes']
            ################################################################################################

            ####<IF TANIMOTO==1, THEN SHOW SAME PREDICTION RESULTS AS THAT OF NCATS SIMILARITY TABLE>####
            pos_class = ['MODERATE', 'HIGH']
            neg_class = ['LOW']
            tanimoto_flag = None
            tanimoto_flag_dict = None

            for key, val in similarity_dict[processed_smiles].items():
                if similarity_dict[processed_smiles][key]['tanimoto'] == 1:
                    tanimoto_flag = True
                    tanimoto_flag_dict = similarity_dict[processed_smiles][key]
                    break

            if tanimoto_flag == True:
                if tanimoto_flag_dict['CPE.ACTIVITY_CLASS'] in neg_class:
                    consensus_prediction_results[processed_smiles]['Activity Model']['prediction'] = 'INACTIVE'
                    consensus_prediction_results[processed_smiles]['Activity Model']['probability'] = '1.0'
                if tanimoto_flag_dict['CPE.ACTIVITY_CLASS'] in pos_class:
                    consensus_prediction_results[processed_smiles]['Activity Model']['prediction'] = 'ACTIVE'
                    consensus_prediction_results[processed_smiles]['Activity Model']['probability'] = '1.0'
                if tanimoto_flag_dict['host_tox_counterscreen.ACTIVITY_CLASS'] in neg_class:
                    consensus_prediction_results[processed_smiles]['Toxicity Model']['prediction'] = 'INACTIVE'
                    consensus_prediction_results[processed_smiles]['Toxicity Model']['probability'] = '1.0'
                if tanimoto_flag_dict['host_tox_counterscreen.ACTIVITY_CLASS'] in pos_class:
                    consensus_prediction_results[processed_smiles]['Toxicity Model']['prediction'] = 'ACTIVE'
                    consensus_prediction_results[processed_smiles]['Toxicity Model']['probability'] = '1.0'
                if tanimoto_flag_dict['ACE2_enzymatic_activity.ACTIVITY_CLASS'] in neg_class:
                    consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model']['prediction'] = 'INACTIVE'
                    consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model']['probability'] = '1.0'
                if tanimoto_flag_dict['ACE2_enzymatic_activity.ACTIVITY_CLASS'] in pos_class:
                    consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model']['prediction'] = 'ACTIVE'
                    consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model']['probability'] = '1.0'
                if tanimoto_flag_dict['AlphaLISA.ACTIVITY_CLASS'] in neg_class:
                    consensus_prediction_results[processed_smiles]['AlphaLisa Model']['prediction'] = 'INACTIVE'
                    consensus_prediction_results[processed_smiles]['AlphaLisa Model']['probability'] = '1.0'
                if tanimoto_flag_dict['AlphaLISA.ACTIVITY_CLASS'] in pos_class:
                    consensus_prediction_results[processed_smiles]['AlphaLisa Model']['prediction'] = 'ACTIVE'
                    consensus_prediction_results[processed_smiles]['AlphaLisa Model']['probability'] = '1.0'
                if tanimoto_flag_dict['AlphaLISA.ACTIVITY_CLASS'] in neg_class:
                    consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model']['prediction'] = 'INACTIVE'
                    consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model']['probability'] = '1.0'
                if tanimoto_flag_dict['AlphaLISA.ACTIVITY_CLASS'] in pos_class:
                    consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model']['prediction'] = 'ACTIVE'
                    consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model']['probability'] = '1.0'

                # NOTE: TruHit labels will be opposite! -->
                if tanimoto_flag_dict['TruHit_Counterscreen.ACTIVITY_CLASS'] in neg_class:
                    consensus_prediction_results[processed_smiles]['TruHit Model']['prediction'] = 'ACTIVE'
                    consensus_prediction_results[processed_smiles]['TruHit Model']['probability'] = '1.0'
                if tanimoto_flag_dict['TruHit_Counterscreen.ACTIVITY_CLASS'] in pos_class:
                    consensus_prediction_results[processed_smiles]['TruHit Model']['prediction'] = 'INACTIVE'
                    consensus_prediction_results[processed_smiles]['TruHit Model']['probability'] = '1.0'
            #############################################################################################


            # Editing predict_dict
            # for key, val in predict_dict.items():
            #     for key2, val2 in val.items():
            #         if val[key2]['prediction'] == 'active' and 'Act' in key2:
            #             predict_dict[key][key2].update({'prediction': 'ACTIVE'})
            #             continue
            #         elif val[key2]['prediction'] == 'inactive' and 'Act' in key2:
            #             predict_dict[key][key2].update({'prediction': 'INACTIVE'})
            #             continue
            #         elif val[key2]['prediction'] == 'active' and 'Tox' in key2:
            #             predict_dict[key][key2].update({'prediction': 'TOXIC'})
            #             continue
            #         elif val[key2]['prediction'] == 'inactive' and 'Tox' in key2:
            #             predict_dict[key][key2].update({'prediction': 'NON-TOXIC'})
            #             continue
            #
            # for key, val in predict_dict.items():
            #     for key2, val2 in val.items():
            #         if key2 == 'ToxTopo':
            #             print("YAYAYAYA")
            #             val['Toxicity-Topological Pharm. Model'] = val.pop(key2)
            #             continue
            #         if key2 == 'ActFP':
            #             val['Activity-Fingerprint Model'] = val.pop(key2)
            #             continue
            #         elif key2 == 'ActDes':
            #             val['Activity-Descriptor Model'] = val.pop(key2)
            #             continue
            #         elif key2 == 'ActTopo':
            #             val['Activity-Topological Pharm. Model'] = val.pop(key2)
            #             continue
            #         if key2 == 'ToxFP':
            #             val['Toxicity-Fingerprint Model'] = val.pop(key2)
            #             continue
            #         elif key2 == 'ToxDes':
            #             print("YAYAYAYA")
            #             val['Toxicity-Descriptor Model'] = val.pop(key2)
            #             continue



            # # Debug 2
            # print('\nPOST EDITING BELOW =>')
            # print('Similairty BELOW =>')
            # pprint.pprint(similarity_dict)
            #
            # print('\nPrediction BELOW =>')
            # pprint.pprint(predict_dict)
            #
            # print('\nALL_DICT BELOW =>')
            # pprint.pprint(all_dict)
            #
            # print('\nCONSENSUS_DICT BELOW =>')
            # pprint.pprint(consensus_prediction_results[processed_smiles])

            if not USE_OCHEM_API:
            #########<OCHEM ALOGPS TOOL CALCULATIONS [NOTE: can only be EXECUTED FROM VIA A LINUX MACHINE!]>#########
            # USE DOCKER FOR BELOW TASK -->
                try:
                    o = OchemToolALOGPS()
                    out, err = o.calculate_alogps(processed_smiles)
                    out, err = str(out), str(err)
                    if 'error' in out or out == '':
                        logp = 'smi_error'
                        logs = 'smi_error'
                    else:
                        s = out
                        logp = s[s.find('logP:') + len('logP:'):s.find('(', s.find('logP:') + len('logP:'))]
                        logs = s[s.find('logS:') + len('logS:'):s.find('(', s.find('logS:') + len('logS:'))]

                    # Debug -->
                    # print('\nlogp-->', logp)
                    # print('\nlogs-->', logs)

                except:
                    logp = 'script_error'
                    logs = 'script_error'
            ######################################

            if USE_OCHEM_API:
            #########<OCHEM ALOGPS API CALCULATIONS>#########
                try:
                    ochem_api_ob = OchemAPIResults()
                    logp = ochem_api_ob.get_ochem_model_results(processed_smiles, 535) # logp
                    logs = ochem_api_ob.get_ochem_model_results(processed_smiles, 536)  # logs

                except:
                    logp = '-'
                    logs = '-'
            ######################################



            ###<FOR RDKIT>=2020 VERSION>###
            # def smi_to_png(smi, query_smi_path, get_binary=False):
            #     mol = Chem.MolFromSmiles(smi)
            #     d2d = rdMolDraw2D.MolDraw2DSVG(300, 300)
            #     d2d.DrawMolecule(mol)
            #     d2d.FinishDrawing()
            #     svg_vector = d2d.GetDrawingText()
            #     cairosvg.svg2png(bytestring=svg_vector, write_to=query_smi_path + 'query_smi.png')
            #
            #     img = Image.open(query_smi_path + 'query_smi.png', mode='r')
            #     img_byte_arr = io.BytesIO()
            #     img.save(img_byte_arr, format='PNG')
            #     encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
            #     return encoded_img

            ###<FOR RDKIT LESS THAN 2020 VERSION>###
            def smi_to_png(smi, query_smi_path, get_binary=False):

                def moltosvg(mol, molSize=(300, 300), kekulize=True):
                    mc = Chem.Mol(mol.ToBinary())
                    if kekulize:
                        try:
                            Chem.Kekulize(mc)
                        except:
                            mc = Chem.Mol(mol.ToBinary())
                    if not mc.GetNumConformers():
                        rdDepictor.Compute2DCoords(mc)
                    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
                    drawer.DrawMolecule(mc)
                    drawer.FinishDrawing()
                    svg = drawer.GetDrawingText()
                    return svg

                mol = Chem.MolFromSmiles(smi)
                svg_vector = moltosvg(mol)
                cairosvg.svg2png(bytestring=svg_vector, write_to=query_smi_path + 'query_smi.png')
                img = Image.open(query_smi_path + 'query_smi.png', mode='r')
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
                return encoded_img

            binary_image = smi_to_png(processed_smiles, './static/images/', get_binary=True)

            # Below tried to preserve ordered dict, but failed
            # with open('ordered_similarity_result.json','w') as f:
            #     json.dump(similarity_dict, f, indent=4)
            #
            # with open('ordered_similarity_result.json','r') as f:
            #     similarity_dict = json.load(f)

            # with open('ordered_similarity_result.pkl','wb') as f:
            #     pickle.dump(similarity_dict, f)
            #
            # with open('ordered_similarity_result.pkl','rb') as f:
            #     similarity_dict = pickle.load(f)

            all_dict['processed_query_smiles'] = processed_smiles
            # Below dict so that sorting is easier at front end, via php
            all_dict['similarity_results'] = similarity_dict[processed_smiles]
            all_dict['three_cl_similarity_results'] = three_cl_similarity_dict[processed_smiles]
            all_dict['prediction_results'] = predict_dict
            all_dict['image'] = binary_image
            all_dict['logp'] = logp
            all_dict['logs'] = logs
            all_dict['molecular_wt'] = molecular_wt
            all_dict['molecular_formula'] = molecular_formula
            all_dict['pubchem_link'] = pubchem_link
            all_dict['pubchem_cid'] = str(pubchem_cid)
            all_dict['drug_central_link'] = drug_central_link
            all_dict['drug_central_id'] = str(drug_central_id)
            all_dict['synonyms'] = synonyms
            all_dict['consensus_prediction_results'] = consensus_prediction_results[processed_smiles]
            ###############<Below Format For easier manipulation at Front-End side>##################
            all_dict['Act_prediction'] = consensus_prediction_results[processed_smiles]['Activity Model']['prediction']
            all_dict['Act_probability'] = consensus_prediction_results[processed_smiles]['Activity Model']['probability']
            all_dict['Tox_prediction'] = consensus_prediction_results[processed_smiles]['Toxicity Model']['prediction']
            all_dict['Tox_probability'] = consensus_prediction_results[processed_smiles]['Toxicity Model']['probability']
            all_dict['AlphaLisa_prediction'] = consensus_prediction_results[processed_smiles]['AlphaLisa Model']['prediction']
            all_dict['AlphaLisa_probability'] = consensus_prediction_results[processed_smiles]['AlphaLisa Model']['probability']
            all_dict['AlphaLisaAndTruHit_prediction'] = consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model']['prediction']
            all_dict['AlphaLisaAndTruHit_probability'] = consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model']['probability']
            all_dict['TruHit_prediction'] = consensus_prediction_results[processed_smiles]['TruHit Model']['prediction']
            all_dict['TruHit_probability'] = consensus_prediction_results[processed_smiles]['TruHit Model']['probability']
            all_dict['ACE2Enzymatic_prediction'] = consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model']['prediction']
            all_dict['ACE2Enzymatic_probability'] = consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model']['probability']
            all_dict['3CLEnzymatic_prediction'] = consensus_prediction_results[processed_smiles]['3CL Enzymatic Model']['prediction']
            all_dict['3CLEnzymatic_probability'] = consensus_prediction_results[processed_smiles]['3CL Enzymatic Model']['probability']
            ##########################################################################################
            all_dict['error'] = {'message': "NO ERROR", 'error': 'None'}
            all_dict['error_flag'] = 'FALSE'

            # pprint.pprint(all_dict['similarity_results'])
            # pprint.pprint(all_dict['similarity_results'])

        except Exception as e:
            all_dict['error'] = {'message': "SCRIPT ERROR", 'error': str(e)}
            all_dict['error_flag'] = 'TRUE'
            # print(all_dict['error'])
            return jsonify(all_dict)


        # Debug 3
        # print('\nFINAL ALL_DICT BELOW =>')
        # pprint.pprint(all_dict['prediction_results'])
        # print("THIS IS IT!", jsonify(all_dict))

        # pprint.pprint(all_dict)
        return jsonify(all_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run(debug=True)
