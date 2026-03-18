from __future__ import annotations


class BusinessRulesService:
    def evaluate(self, structured_data: dict, text: str) -> dict:
        issues: list[dict[str, str]] = []
        recommendations: list[str] = []

        if len(text.strip()) < 80:
            issues.append({"code": "TEXT_TOO_SHORT", "message": "Le texte extrait semble trop court pour une analyse fiable."})

        if structured_data.get("document_type") in (None, "", "unknown"):
            issues.append({"code": "DOC_TYPE_MISSING", "message": "Le type de document n'a pas ete determine."})

        if not structured_data.get("summary"):
            issues.append({"code": "SUMMARY_MISSING", "message": "Aucun resume n'a ete produit."})

        if structured_data.get("confidence", 0) < 0.5:
            issues.append({"code": "LOW_CONFIDENCE", "message": "La confiance de l'extraction structuree est faible."})

        financial_data = structured_data.get("financial_data", {})
        if structured_data.get("document_type") == "financial_statement":
            if not financial_data.get("statement_sections_detected"):
                issues.append({"code": "FS_SECTIONS_MISSING", "message": "Les sections des etats financiers n'ont pas ete detectees."})
            if not any(financial_data.get("key_metrics", {}).values()):
                issues.append({"code": "FS_KEY_METRICS_MISSING", "message": "Aucun poste financier cle n'a ete extrait."})
                recommendations.append("Activer OCR ou fournir un fichier Excel/CSV source pour ameliorer la precision.")
            if not financial_data.get("risk_indicators", {}).get("coverage_ready"):
                recommendations.append("Completer les montants d'actif, passif, revenus et charges pour calculer les ratios de risque.")

        iq_solution_fit = structured_data.get("iq_solution_fit", {})
        if iq_solution_fit:
            fit_score = iq_solution_fit.get("fit_score", 0)
            if fit_score < 70:
                recommendations.append("Renforcer le mapping des postes financiers et la classification documentaire pour mieux repondre au besoin IQ.")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "score": max(0, 100 - len(issues) * 20),
            "recommendations": recommendations,
        }
