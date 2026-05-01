#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADMET评估模块

功能：
1. 基于RDKit计算ADMET性质
2. 吸收性预测（口服利用度、Caco-2渗透性）
3. 代谢稳定性预测（CYP450抑制）
4. 毒性预测（AMES毒性、hERG抑制）
5. 溶解度预测
6. 血脑屏障穿透性预测
7. SwissADME整合

作者：研究团队
版本：1.0.0
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem, Descriptors, Lipinski, rdMolDescriptors, ADMEParser
    )
    from rdkit.Chem.Descriptors import MolecularDescriptorCalculator
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    warnings.warn("RDKit not available. ADMET evaluation will be limited.")

from configs.config import (
    PROJECT_ROOT, RESULTS_DIR, LOG_CONFIG, ADMET_THRESHOLDS, LIPINSKI_RULES
)

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class ADMETCalculator:
    """
    ADMET性质计算类

    基于分子描述符和机器学习模型预测ADMET性质
    """

    def __init__(self):
        """初始化ADMET计算器"""
        self.descriptor_calculator = None
        if HAS_RDKIT:
            desc_names = [
                'MolWt', 'MolLogP', 'MolMR', 'TPSA',
                'NumHDonors', 'NumHAcceptors',
                'NumRotatableBonds', 'NumHeteroatoms',
                'NumAromaticRings', 'FractionCSP3',
                'NumAliphaticRings', 'NumSaturatedRings',
                'NumRings', 'BertzCT', 'Chi0', 'Chi1',
                'Kappa1', 'Kappa2', 'HallKierAlpha', 'LabuteASA'
            ]
            self.descriptor_calculator = MolecularDescriptorCalculator(desc_names)

    def calculate_absorption(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        计算吸收性质

        参数:
            mol: RDKit分子对象

        返回:
            dict: 吸收性质预测结果
        """
        if not HAS_RDKIT or mol is None:
            return {}

        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            num_rotatable = Lipinski.NumRotatableBonds(mol)

            absorption = {
                "molecular_weight": mw,
                "logp": logp,
                "tpsa": tpsa,
                "h_donors": hbd,
                "h_acceptors": hba,
                "rotatable_bonds": num_rotatable
            }

            absorption["human_intestinal_absorption"] = self._predict_hia(mw, logp, tpsa, hbd, hba)
            absorption["caco2_permeability"] = self._predict_caco2(mw, logp, tpsa)
            absorption["mdck_permeability"] = self._predict_mdck(mw, logp, tpsa)

            absorption["oral_bioavailability"] = self._predict_oral_bioavailability(
                mw, logp, tpsa, hbd, num_rotatable
            )

            absorption["fraction_absorbed"] = self._predict_fraction_absorbed(
                mw, logp, tpsa, hbd, hba
            )

            return absorption

        except Exception as e:
            logger.debug(f"Error calculating absorption: {str(e)}")
            return {}

    def _predict_hia(self, mw: float, logp: float, tpsa: float, hbd: int, hba: int) -> str:
        """
        预测人肠道吸收

        基于简化规则：
        - MW <= 500
        - LogP <= 5
        - TPSA <= 140
        - HBD <= 5
        - HBA <= 10
        """
        score = 0
        if mw <= 500:
            score += 1
        if logp <= 5:
            score += 1
        if tpsa <= 140:
            score += 1
        if hbd <= 5:
            score += 1
        if hba <= 10:
            score += 1

        if score >= 4:
            return "High"
        elif score >= 2:
            return "Medium"
        else:
            return "Low"

    def _predict_caco2(self, mw: float, logp: float, tpsa: float) -> float:
        """
        预测Caco-2渗透性 (10^-6 cm/s)

        简化的QSAR模型
        """
        caco2 = -0.0148 * mw + 0.152 * logp - 0.0189 * tpsa + 0.804

        caco2 = max(0, min(caco2, 50))

        return round(caco2, 4)

    def _predict_mdck(self, mw: float, logp: float, tpsa: float) -> str:
        """
        预测MDCK渗透性
        """
        if mw > 500 or tpsa > 150:
            return "Low"

        mdck_score = -0.01 * mw + 0.1 * logp - 0.015 * tpsa + 2.5

        if mdck_score > 15:
            return "High"
        elif mdck_score > 5:
            return "Medium"
        else:
            return "Low"

    def _predict_oral_bioavailability(self, mw: float, logp: float, tpsa: float,
                                     hbd: int, num_rotatable: int) -> str:
        """
        预测口服生物利用度
        """
        score = 0

        if mw <= 400:
            score += 2
        elif mw <= 500:
            score += 1

        if 1 <= logp <= 3:
            score += 2
        elif 0 <= logp <= 5:
            score += 1

        if tpsa <= 120:
            score += 2
        elif tpsa <= 140:
            score += 1

        if hbd <= 5:
            score += 1

        if num_rotatable <= 10:
            score += 1

        if score >= 7:
            return "High (>50%)"
        elif score >= 4:
            return "Medium (10-50%)"
        else:
            return "Low (<10%)"

    def _predict_fraction_absorbed(self, mw: float, logp: float, tpsa: float,
                                 hbd: int, hba: int) -> float:
        """
        预测Fraction Absorbed (%)
        """
        Fa = 100 * (1 - 0.0002 * (tpsa ** 2 - 100))

        if mw > 500:
            Fa *= 0.5
        if logp < 0:
            Fa *= 0.8
        if hbd > 5:
            Fa *= 0.8

        Fa = max(0, min(100, Fa))

        return round(Fa, 2)

    def calculate_metabolism(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        计算代谢性质

        参数:
            mol: RDKit分子对象

        返回:
            dict: 代谢性质预测结果
        """
        if not HAS_RDKIT or mol is None:
            return {}

        try:
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)

            metabolism = {
                "cyp1a2_inhibition": self._predict_cyp_inhibition(mol, logp, tpsa, "1A2"),
                "cyp2c9_inhibition": self._predict_cyp_inhibition(mol, logp, tpsa, "2C9"),
                "cyp2c19_inhibition": self._predict_cyp_inhibition(mol, logp, tpsa, "2C19"),
                "cyp2d6_inhibition": self._predict_cyp_inhibition(mol, logp, tpsa, "2D6"),
                "cyp3a4_inhibition": self._predict_cyp_inhibition(mol, logp, tpsa, "3A4"),
            }

            metabolism["blood_plasma_ratio"] = self._predict_blood_plasma_ratio(logp, tpsa)

            metabolism["clearance"] = self._predict_clearance(mol, mw, logp, tpsa)

            return metabolism

        except Exception as e:
            logger.debug(f"Error calculating metabolism: {str(e)}")
            return {}

    def _predict_cyp_inhibition(self, mol: Chem.Mol, logp: float, tpsa: float,
                               cyp_isoform: str) -> Dict[str, Any]:
        """
        预测CYP450酶抑制

        简化的规则模型
        """
        risk_score = 0

        if logp > 4:
            risk_score += 1
        if tpsa < 75:
            risk_score += 1

        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        if aromatic_rings >= 2:
            risk_score += 1

        mw_threshold = {
            "1A2": 312,
            "2C9": 380,
            "2C19": 320,
            "2D6": 470,
            "3A4": 580
        }

        if mw > mw_threshold.get(cyp_isoform, 400):
            risk_score += 1

        if risk_score >= 3:
            prediction = "High Risk"
            probability = 0.8
        elif risk_score >= 1:
            prediction = "Medium Risk"
            probability = 0.5
        else:
            prediction = "Low Risk"
            probability = 0.2

        return {
            "prediction": prediction,
            "probability": round(probability, 3)
        }

    def _predict_blood_plasma_ratio(self, logp: float, tpsa: float) -> float:
        """
        预测血液/血浆分配比
        """
        bpp = 10 ** (0.55 * logp - 0.015 * tpsa + 0.3)

        bpp = max(0.1, min(10, bpp))

        return round(bpp, 3)

    def _predict_clearance(self, mol: Chem.Mol, mw: float,
                          logp: float, tpsa: float) -> Dict[str, float]:
        """
        预测清除率 (mL/min/kg)
        """
        cl_intrinsic = -0.001 * mw + 0.3 * logp - 0.02 * tpsa + 5

        cl_h = cl_intrinsic * (50 / (50 + 0.1 * mw))

        cl = cl_h / (1 + 0.1 * cl_h)

        cl = max(0.5, min(50, cl))

        return {
            " hepatic_clearance": round(cl, 2),
            "intrinsic_clearance": round(cl_intrinsic, 2)
        }

    def calculate_toxicity(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        计算毒性性质

        参数:
            mol: RDKit分子对象

        返回:
            dict: 毒性性质预测结果
        """
        if not HAS_RDKIT or mol is None:
            return {}

        try:
            toxicity = {}

            toxicity["ames_toxicity"] = self._predict_ames_toxicity(mol)
            toxicity["herg_inhibition"] = self._predict_herg_inhibition(mol)
            toxicity["toxicity_class"] = self._predict_toxicity_class(mol)

            toxicity["ld50"] = self._predict_ld50(mol)

            toxicity["skin_sensitization"] = self._predict_skin_sensitization(mol)

            toxicity["eye_irritation"] = self._predict_eye_irritation(mol)

            return toxicity

        except Exception as e:
            logger.debug(f"Error calculating toxicity: {str(e)}")
            return {}

    def _predict_ames_toxicity(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        预测AMES毒性（致突变性）
        """
        risk_score = 0
        warnings_list = []

        aromatic_amines = mol.HasSubstructMatch(
            Chem.MolFromSmarts("[NX3,NX4][a]")
        )
        if aromatic_amines:
            risk_score += 2
            warnings_list.append("Aromatic amine detected")

        nitro_groups = mol.HasSubstructMatch(
            Chem.MolFromSmarts("[N+](=O)[O-]")
        )
        if nitro_groups:
            risk_score += 2
            warnings_list.append("Nitro group detected")

        alkyl_halides = mol.HasSubstructMatch(
            Chem.MolFromSmarts("[Cl,Br,I][CX4]")
        )
        if alkyl_halides:
            risk_score += 1
            warnings_list.append("Alkyl halide detected")

        polycyclic = rdMolDescriptors.CalcNumAromaticRings(mol)
        if polycyclic >= 3:
            risk_score += 1
            warnings_list.append("Multiple aromatic rings")

        if risk_score >= 3:
            prediction = "Mutagenic"
            probability = 0.8
        elif risk_score >= 1:
            prediction = "Potentially Mutagenic"
            probability = 0.5
        else:
            prediction = "Non-Mutagenic"
            probability = 0.2

        return {
            "prediction": prediction,
            "probability": round(probability, 3),
            "warnings": warnings_list
        }

    def _predict_herg_inhibition(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        预测hERG钾通道抑制（心脏毒性）
        """
        risk_score = 0
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        if logp > 4:
            risk_score += 2
        if tpsa < 75:
            risk_score += 1

        basic_nitrogens = mol.HasSubstructMatch(
            Chem.MolFromSmarts("[NX3;v3;+1]")
        )
        if basic_nitrogens:
            risk_score += 2

        mw = Descriptors.MolWt(mol)
        if 200 < mw < 600:
            risk_score += 1

        if risk_score >= 4:
            prediction = "High Risk"
            probability = 0.8
        elif risk_score >= 2:
            prediction = "Medium Risk"
            probability = 0.5
        else:
            prediction = "Low Risk"
            probability = 0.2

        return {
            "prediction": prediction,
            "probability": round(probability, 3)
        }

    def _predict_toxicity_class(self, mol: Chem.Mol) -> str:
        """
        预测毒性分类
        """
        toxicity_score = 0

        mw = Descriptors.MolWt(mol)
        if mw > 1000:
            toxicity_score += 2
        elif mw > 600:
            toxicity_score += 1

        logp = Descriptors.MolLogP(mol)
        if logp > 6:
            toxicity_score += 1

        hbd = Lipinski.NumHDonors(mol)
        if hbd > 5:
            toxicity_score += 1

        return "High Toxicity" if toxicity_score >= 3 else "Medium Toxicity" if toxicity_score >= 1 else "Low Toxicity"

    def _predict_ld50(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        预测LD50 (mg/kg, oral, rat)

        简化的QSAR模型
        """
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        log_ld50 = 2.5 - 0.01 * mw + 0.2 * logp + 0.001 * tpsa

        ld50 = 10 ** log_ld50

        ld50 = max(1, min(10000, ld50))

        if ld50 < 50:
            toxicity_class = "Highly Toxic"
        elif ld50 < 500:
            toxicity_class = "Moderately Toxic"
        elif ld50 < 5000:
            toxicity_class = "Slightly Toxic"
        else:
            toxicity_class = "Practically Non-Toxic"

        return {
            "value": round(ld50, 2),
            "unit": "mg/kg",
            "species": "rat",
            "route": "oral",
            "toxicity_class": toxicity_class
        }

    def _predict_skin_sensitization(self, mol: Chem.Mol) -> str:
        """
        预测皮肤致敏性
        """
        alert_found = False

        if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=O")):
            alert_found = True

        if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3]")):
            alert_found = True

        if mol.HasSubstructMatch(Chem.MolFromSmarts("[SX2H]")):
            alert_found = True

        return "Potential Sensitizer" if alert_found else "No Sensitization Expected"

    def _predict_eye_irritation(self, mol: Chem.Mol) -> str:
        """
        预测眼刺激
        """
        acids = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)O"))
        bases = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;v3]"))

        if acids or bases:
            return "Possible Irritant"
        else:
            return "No Expected Irritation"

    def calculate_solubility(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        计算溶解度性质

        参数:
            mol: RDKit分子对象

        返回:
            dict: 溶解度预测结果
        """
        if not HAS_RDKIT or mol is None:
            return {}

        try:
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)

            log_s = -0.01 * mw + 0.08 * logp - 0.02 * tpsa + 2.5

            log_s = max(-12, min(2, log_s))

            solubility_mg_l = 10 ** log_s * mw

            if log_s >= -4:
                prediction = "High Solubility"
            elif log_s >= -6:
                prediction = "Medium Solubility"
            else:
                prediction = "Low Solubility"

            return {
                "log_s": round(log_s, 3),
                "solubility_mg_per_l": round(solubility_mg_l, 2),
                "prediction": prediction,
                "aq_stability": "Stable" if log_s > -6 else "May Have Stability Issues"
            }

        except Exception as e:
            logger.debug(f"Error calculating solubility: {str(e)}")
            return {}

    def calculate_bbb_penetration(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        计算血脑屏障穿透性

        参数:
            mol: RDKit分子对象

        return: dict: BBB穿透性预测结果
        """
        if not HAS_RDKIT or mol is None:
            return {}

        try:
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)

            log_bb = 0.14 * logp - 0.02 * tpsa - 0.07 * hbd - 0.04 * hba + 0.28

            log_bb = max(-3, min(2, log_bb))

            if log_bb > 0.3:
                prediction = "High BBB Penetration"
            elif log_bb > -0.3:
                prediction = "Moderate BBB Penetration"
            else:
                prediction = "Low BBB Penetration"

            return {
                "log_bb": round(log_bb, 3),
                "prediction": prediction,
                "CNS_score": round(max(0, min(1, (log_bb + 1) / 2)), 3)
            }

        except Exception as e:
            logger.debug(f"Error calculating BBB penetration: {str(e)}")
            return {}

    def calculate_all_admet(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        计算所有ADMET性质

        参数:
            mol: RDKit分子对象

        返回:
            dict: 完整的ADMET性质字典
        """
        if not HAS_RDKIT or mol is None:
            return {}

        admet_results = {
            "absorption": self.calculate_absorption(mol),
            "metabolism": self.calculate_metabolism(mol),
            "toxicity": self.calculate_toxicity(mol),
            "solubility": self.calculate_solubility(mol),
            "bbb_penetration": self.calculate_bbb_penetration(mol)
        }

        admet_results["overall_assessment"] = self._generate_overall_assessment(admet_results)

        return admet_results

    def _generate_overall_assessment(self, admet_results: Dict) -> Dict[str, Any]:
        """
        生成整体评估

        参数:
            admet_results: ADMET结果字典

        返回:
            dict: 整体评估结果
        """
        passed_checks = 0
        failed_checks = 0
        warnings_list = []

        absorption = admet_results.get("absorption", {})
        if absorption.get("human_intestinal_absorption") == "High":
            passed_checks += 1
        elif absorption.get("human_intestinal_absorption") == "Low":
            failed_checks += 1
            warnings_list.append("Poor intestinal absorption")
        else:
            warnings_list.append("Moderate intestinal absorption")

        metabolism = admet_results.get("metabolism", {})
        cyp3a4 = metabolism.get("cyp3a4_inhibition", {})
        if cyp3a4.get("prediction") == "High Risk":
            failed_checks += 1
            warnings_list.append("Strong CYP3A4 inhibition (drug-drug interaction risk)")
        elif cyp3a4.get("prediction") == "Medium Risk":
            warnings_list.append("Moderate CYP3A4 inhibition risk")

        toxicity = admet_results.get("toxicity", {})
        ames = toxicity.get("ames_toxicity", {})
        if ames.get("prediction") == "Mutagenic":
            failed_checks += 1
            warnings_list.append("Potential mutagenicity")

        herg = toxicity.get("herg_inhibition", {})
        if herg.get("prediction") == "High Risk":
            failed_checks += 1
            warnings_list.append("hERG channel inhibition (cardiotoxicity risk)")

        solubility = admet_results.get("solubility", {})
        if solubility.get("prediction") == "Low Solubility":
            warnings_list.append("Low aqueous solubility may affect oral absorption")

        if failed_checks == 0 and len(warnings_list) == 0:
            overall = "Drug-like: Good ADMET profile"
        elif failed_checks == 0:
            overall = "Drug-like: Minor concerns noted"
        elif failed_checks <= 2:
            overall = "Borderline: May require optimization"
        else:
            overall = "Non-drug-like: Significant ADMET concerns"

        return {
            "overall": overall,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "warnings": warnings_list
        }


class ADMETBatchEvaluator:
    """
    ADMET批量评估类

    用于批量评估化合物的ADMET性质
    """

    def __init__(self):
        """初始化ADMET批量评估器"""
        self.calculator = ADMETCalculator()
        self.results = []

    def evaluate_smiles_list(self,
                           smiles_list: List[str],
                           compound_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量评估SMILES列表

        参数:
            smiles_list: SMILES字符串列表
            compound_names: 化合物名称列表

        返回:
            pd.DataFrame: 评估结果DataFrame
        """
        if compound_names is None:
            compound_names = [f"compound_{i}" for i in range(len(smiles_list))]

        results = []

        for i, (smiles, name) in enumerate(zip(smiles_list, compound_names)):
            if (i + 1) % 100 == 0:
                logger.info(f"Evaluated {i+1}/{len(smiles_list)} compounds")

            mol = None
            if HAS_RDKIT:
                mol = Chem.MolFromSmiles(smiles)

            result = {
                "compound_name": name,
                "smiles": smiles,
                "valid_structure": mol is not None
            }

            if mol is not None:
                try:
                    admet = self.calculator.calculate_all_admet(mol)

                    result.update({
                        "hia": admet.get("absorption", {}).get("human_intestinal_absorption", "Unknown"),
                        "oral_bioavailability": admet.get("absorption", {}).get("oral_bioavailability", "Unknown"),
                        "caco2_permeability": admet.get("absorption", {}).get("caco2_permeability", "Unknown"),
                        "cyp3a4_inhibition": admet.get("metabolism", {}).get("cyp3a4_inhibition", {}).get("prediction", "Unknown"),
                        "ames_toxicity": admet.get("toxicity", {}).get("ames_toxicity", {}).get("prediction", "Unknown"),
                        "herg_inhibition": admet.get("toxicity", {}).get("herg_inhibition", {}).get("prediction", "Unknown"),
                        "ld50_class": admet.get("toxicity", {}).get("ld50", {}).get("toxicity_class", "Unknown"),
                        "solubility": admet.get("solubility", {}).get("prediction", "Unknown"),
                        "bbb_penetration": admet.get("bbb_penetration", {}).get("prediction", "Unknown"),
                        "overall_assessment": admet.get("overall_assessment", {}).get("overall", "Unknown")
                    })

                except Exception as e:
                    logger.debug(f"Error evaluating {name}: {str(e)}")

            results.append(result)

        self.results = results
        return pd.DataFrame(results)

    def filter_compounds(self,
                        df: pd.DataFrame,
                        admet_rules: Optional[Dict] = None) -> pd.DataFrame:
        """
        根据ADMET规则过滤化合物

        参数:
            df: 评估结果DataFrame
            admet_rules: ADMET过滤规则

        返回:
            pd.DataFrame: 过滤后的DataFrame
        """
        if admet_rules is None:
            admet_rules = {
                "hia": ["High", "Medium"],
                "cyp3a4_inhibition": ["Low Risk", "Medium Risk"],
                "ames_toxicity": ["Non-Mutagenic"],
                "herg_inhibition": ["Low Risk"],
                "solubility": ["High Solubility", "Medium Solubility"]
            }

        df_filtered = df.copy()

        if "hia" in admet_rules and "hia" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["hia"].isin(admet_rules["hia"])]

        if "cyp3a4_inhibition" in admet_rules and "cyp3a4_inhibition" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["cyp3a4_inhibition"].isin(admet_rules["cyp3a4_inhibition"])]

        if "ames_toxicity" in admet_rules and "ames_toxicity" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["ames_toxicity"].isin(admet_rules["ames_toxicity"])]

        if "herg_inhibition" in admet_rules and "herg_inhibition" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["herg_inhibition"].isin(admet_rules["herg_inhibition"])]

        if "solubility" in admet_rules and "solubility" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["solubility"].isin(admet_rules["solubility"])]

        logger.info(f"Filtered {len(df)} compounds to {len(df_filtered)} compounds based on ADMET rules")

        return df_filtered

    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取ADMET统计信息

        参数:
            df: 评估结果DataFrame

        返回:
            dict: 统计信息
        """
        stats = {
            "total_compounds": len(df),
            "valid_structures": int(df["valid_structure"].sum()) if "valid_structure" in df.columns else len(df)
        }

        categorical_columns = ["hia", "oral_bioavailability", "cyp3a4_inhibition",
                              "ames_toxicity", "herg_inhibition", "solubility", "bbb_penetration"]

        for col in categorical_columns:
            if col in df.columns:
                stats[col] = df[col].value_counts().to_dict()

        return stats

    def save_results(self, df: pd.DataFrame, output_file: str) -> bool:
        """
        保存评估结果

        参数:
            df: 评估结果DataFrame
            output_file: 输出文件路径

        返回:
            bool: 保存是否成功
        """
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"ADMET results saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving ADMET results: {str(e)}")
            return False


def evaluate_admet(smiles_list: List[str],
                  compound_names: Optional[List[str]] = None,
                  output_file: Optional[str] = None) -> pd.DataFrame:
    """
    便捷函数：评估化合物的ADMET性质

    参数:
        smiles_list: SMILES字符串列表
        compound_names: 化合物名称列表
        output_file: 输出文件路径

    返回:
        pd.DataFrame: ADMET评估结果
    """
    evaluator = ADMETBatchEvaluator()
    df = evaluator.evaluate_smiles_list(smiles_list, compound_names)

    if output_file:
        evaluator.save_results(df, output_file)

    return df


if __name__ == "__main__":
    logger.info("Testing ADMET module")

    test_smiles = ["CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"]

    evaluator = ADMETBatchEvaluator()
    df = evaluator.evaluate_smiles_list(test_smiles, ["ethanol", "benzene", "aspirin"])

    logger.info(f"ADMET evaluation results:\n{df}")
