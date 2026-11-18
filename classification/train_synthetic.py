"""Synthetic Bank Transaction Dataset Generator and Model Trainer.

Generates 200+ realistic, noisy bank transaction descriptions across 8 standard
financial categories and trains a scikit-learn TF-IDF + Classifier pipeline
without using any external LLMs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Ensure project root is in sys.path when executed directly as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.pipeline import Pipeline

from config.settings import CLASSIFICATION_DIR

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = CLASSIFICATION_DIR / "saved_models"
DEFAULT_MODEL_PATH = SAVED_MODELS_DIR / "classifier.joblib"

# Base templates and merchant fragments for 8 standard categories
CATEGORY_TEMPLATES = {
    "Salary / Income": [
        "DIRECT DEP ACME CORP PAYROLL",
        "SALARY CREDIT FOR MONTH OF JAN",
        "TECH ENTERPRISES DIRECT DEPOSIT SALARY",
        "PAYROLL DISBURSEMENT WAGES",
        "STIPEND FELLOWSHIP ADVANCE",
        "GLOBAL SERVICES MONTHLY SALARY",
        "BONUS PAYMENT ANNUAL PERFORMANCE",
        "CONSULTING RETAINER SALARY CREDIT",
        "DIRECT DEPOSIT GOVT WAGES",
        "MONTHLY SALARY TRANSFER REF #88291",
        "INNOVATION LABS SALARY PAYMENT",
        "EMPLOYER PAYROLL ADVANCE DEP",
        "WEEKLY WAGES DISBURSEMENT",
        "EXECUTIVE PAYROLL DEPOSIT",
        "ACH CREDIT SALARY SETTLEMENT",
    ],
    "Utilities & Bills": [
        "PACIFIC GAS & ELECTRIC UTILITY BILL",
        "CITY WATER & SEWER DEPT PAYMENT",
        "CON EDISON ELECTRICITY AUTO-PAY",
        "VERIZON WIRELESS MONTHLY PHONE BILL",
        "AT&T BROADBAND INTERNET SERVICE",
        "COMCAST XFINITY CABLE INTERNET",
        "T-MOBILE CELLULAR RECHARGE ONLINE",
        "VODAFONE PREPAID MOBILE RECHARGE",
        "MUNICIPAL WATER SUPPLY BILL",
        "SUNSHINE POWER ELECTRIC UTILITIES",
        "NATIONAL GRID NATURAL GAS AUTO-DEBIT",
        "METRO BROADBAND FIBER BILL",
        "SPECTRUM CABLE & WIFI PAYMENT",
        "STATE ELECTRICITY BOARD BILL",
        "AIRTEL DTH & BROADBAND RECHARGE",
    ],
    "Groceries & Supermarket": [
        "WALMART SUPERCENTER STORE #1042",
        "WHOLE FOODS MARKET ORGANIC GROCERY",
        "TRADER JOE'S NEIGHBORHOOD GROCERY",
        "SAFEWAY SUPERMARKET #882",
        "KROGER FOOD AND PHARMACY GROCERIES",
        "COSTCO WHOLESALE BULK GROCERY",
        "ALDI GROCERY MART PURCHASE",
        "TARGET GROCERY & ESSENTIALS",
        "SPROUTS FARMERS MARKET PRODUCE",
        "RELIANCE FRESH GROCERY STORE",
        "BIG BAZAAR SUPERMARKET",
        "LOCAL CORNER GROCERY STORE",
        "FOOD LION SUPERMARKET #402",
        "HEB GROCERY STORE AUSTIN",
        "MARKET BASKET SUPERMARKET",
    ],
    "Dining & Food": [
        "STARBUCKS STORE #342 CAFE COFFEE",
        "MCDONALD'S DRIVE-THRU BURGER #992",
        "CHIPOTLE MEXICAN GRILL RESTAURANT",
        "UBER EATS ONLINE FOOD DELIVERY",
        "DOORDASH RESTAURANT MEAL ORDER",
        "SWIGGY FOOD ORDER DINING",
        "ZOMATO RESTAURANT TAKEAWAY",
        "DOMINO'S PIZZA STORE DELIVERY",
        "PANERA BREAD BAKERY CAFE",
        "DUNKIN DONUTS COFFEE AND BREAKFAST",
        "SUBWAY SANDWICHES #1209",
        "TACO BELL RESTAURANT #448",
        "THE CHEESECAKE FACTORY DINING",
        "LOCAL ITALIAN BISTRO DINNER",
        "BLUE BOTTLE COFFEE CAFE",
    ],
    "Transfers & UPI": [
        "UPI / 402918239102 / PAYMENT TO VENDOR",
        "NEFT OUTWARD TRANSFER TO ALICE WALKER",
        "IMPS P2P FUNDS TRANSFER TO BOB",
        "ZELLE TRANSFER SENT TO JOHN DOE",
        "VENMO PAYMENT CASHOUT TO BANK",
        "RTGS SETTLEMENT FUNDS TRANSFER",
        "GOOGLE PAY UPI TRANSFER TO SHOP",
        "PHONEPE UPI PAYMENT TO MERCHANT",
        "WIRE TRANSFER OUTWARD REF #8829",
        "PAYPAL TRANSFER TO CHECKING ACCOUNT",
        "INTER-ACCOUNT FUNDS TRANSFER",
        "ONLINE BANKING TRANSFER TO SAVINGS",
        "CASH APP INSTANT CASHOUT TRANSFER",
        "P2P SETTLEMENT TO COLLEAGUE",
        "DOMESTIC WIRE TRANSFER CLEARING",
    ],
    "ATM & Cash": [
        "ATM CASH WITHDRAWAL DOWNTOWN BRANCH",
        "CHASE ATM CASH DISPENSE #4402",
        "BANK OF AMERICA ATM CASH W/D",
        "WELLS FARGO ATM WITHDRAWAL CASH",
        "ATM SURCHARGE WITHDRAWAL MAIN ST",
        "CASH WITHDRAWAL BRANCH TELLER",
        "CITIBANK ATM CASH MACHINE",
        "OFFSHORE ATM CASH DISBURSEMENT",
        "NON-NETWORK ATM CASH WITHDRAWAL",
        "AIRPORT ATM TERMINAL CASH W/D",
        "METRO STATION ATM CASH WITHDRAWAL",
        "CONVENIENCE STORE ATM DISPENSE",
        "DRIVE-UP ATM CASH WITHDRAWAL",
        "RETAIL STORE ATM CASH OUT",
        "CAMPUS ATM CASH WITHDRAWAL #12",
    ],
    "Investment & Insurance": [
        "ZERODHA BROKING MUTUAL FUND SIP",
        "VANGUARD INDEX FUND INVESTMENT",
        "FIDELITY BROKERAGE STOCK PURCHASE",
        "LIC LIFE INSURANCE ANNUAL PREMIUM",
        "GEICO GENERAL AUTO INSURANCE POLICY",
        "STATE FARM HOME INSURANCE AUTO-DEBIT",
        "CHARLES SCHWAB BROKERAGE ACCOUNT",
        "ROBINHOOD SECURITIES INVESTMENT",
        "GROWW MUTUAL FUND MONTHLY SIP",
        "PRUDENTIAL RETIREMENT SAVINGS PLAN",
        "HEALTH INSURANCE PREMIUM AUTO-PAY",
        "HDFC LIFE INSURANCE POLICY PAYMENT",
        "ETRADE FINANCIAL STOCK ORDER",
        "VANGUARD 401K RETIREMENT DEPOSIT",
        "MUTUAL FUND SYSTEMATIC INVESTMENT",
    ],
    "Entertainment & Subscriptions": [
        "NETFLIX STREAMING MONTHLY SUBSCRIPTION",
        "SPOTIFY MUSIC FAMILY PLAN AUTO-DEBIT",
        "AMAZON PRIME ANNUAL MEMBERSHIP",
        "DISNEY PLUS ENTERTAINMENT STREAMING",
        "APPLE SERVICES ICLOUD AND MUSIC",
        "HULU PLUS LIVE TV SUBSCRIPTION",
        "YOUTUBE PREMIUM FAMILY MEMBERSHIP",
        "HBO MAX STREAMING ENTERTAINMENT",
        "AMC THEATRES CINEMA TICKETS",
        "REGAL CINEMAS MOVIE TICKETS",
        "PLAYSTATION NETWORK STORE PURCHASE",
        "XBOX LIVE GAME PASS SUBSCRIPTION",
        "AUDIBLE AUDIOBOOK MONTHLY SUB",
        "NEW YORK TIMES DIGITAL SUBSCRIPTION",
        "PATREON MONTHLY CREATOR MEMBERSHIP",
    ],
}

# Modifiers for synthetic noise and variation
PREFIXES = ["POS", "ACH", "DEBIT", "VISA", "CARD", "ONLINE", "CHECKCARD", "PURCHASE"]
CITIES = ["NEW YORK NY", "SAN JOSE CA", "CHICAGO IL", "AUSTIN TX", "LONDON UK", "MUMBAI IN", "SEATTLE WA", "BOSTON MA"]


def generate_synthetic_training_data(multiplier: int = 2) -> pd.DataFrame:
    """Generate 200+ synthetic labeled transaction descriptions.

    Args:
        multiplier: Multiplier factor across base templates (default: 2 yields ~250+ rows).

    Returns:
        pd.DataFrame with 'description' and 'category' columns.
    """
    records = []

    for category, templates in CATEGORY_TEMPLATES.items():
        # Add base templates
        for t in templates:
            records.append({"description": t, "category": category})

        # Add variations with prefix and city noise
        for i in range(multiplier):
            for idx, t in enumerate(templates):
                prefix = PREFIXES[(idx + i) % len(PREFIXES)]
                city = CITIES[(idx + i) % len(CITIES)]
                ref_num = (idx * 31 + i * 17) % 9999 + 1000

                # Formats
                if i % 2 == 0:
                    noisy_desc = f"{prefix} * {t} #{ref_num} {city}"
                else:
                    noisy_desc = f"{t} REF-{ref_num} {prefix}"

                records.append({"description": noisy_desc, "category": category})

    df = pd.DataFrame(records).drop_duplicates(subset=["description"]).reset_index(drop=True)
    return df


def train_synthetic_model(save_path: Optional[Union[str, Path]] = None) -> Pipeline:
    """Generate synthetic bank transaction dataset and train the ML classifier pipeline.

    Args:
        save_path: Destination path for joblib model file.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    dest = Path(save_path) if save_path else DEFAULT_MODEL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic training dataset...")
    df_train = generate_synthetic_training_data(multiplier=2)
    logger.info("Generated %d synthetic labeled training transactions across %d categories.", len(df_train), df_train["category"].nunique())

    from classification.ml_classifier import TraditionalMLClassifier

    classifier = TraditionalMLClassifier(model_path=dest, auto_bootstrap=False)
    pipeline = classifier.train(df_train, model_path=dest)
    logger.info("Synthetic training complete! Model saved to %s", dest)

    return pipeline


if __name__ == "__main__":
    train_synthetic_model()
