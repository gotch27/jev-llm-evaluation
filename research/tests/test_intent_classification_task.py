from dataclasses import replace
from pathlib import Path

import pytest
from typesafe_sdk import Choice

from thesis_research.config import load_experiment_config
from thesis_research.tasks import build_banking77_task

CONFIG_PATH = Path(__file__).parents[1] / "experiments" / "intent_classification" / "banking77.toml"
INSTRUCTIONS = (
    "Classify the supplied customer message into the single listed banking intent that best "
    "matches the customer’s primary request or problem. Use the banking feature or transaction "
    "type and its stated status, such as pending, declined, reverted, missing, unrecognized, or "
    "charged a fee, to distinguish similar intents. Use only information in the customer message "
    "and do not infer unstated events. Every message belongs to one listed intent; when wording is "
    "brief or ambiguous, select the closest supported intent."
)
LABELS = [
    "card_arrival",
    "card_linking",
    "exchange_rate",
    "card_payment_wrong_exchange_rate",
    "extra_charge_on_statement",
    "pending_cash_withdrawal",
    "fiat_currency_support",
    "card_delivery_estimate",
    "automatic_top_up",
    "card_not_working",
    "exchange_via_app",
    "lost_or_stolen_card",
    "age_limit",
    "pin_blocked",
    "contactless_not_working",
    "top_up_by_bank_transfer_charge",
    "pending_top_up",
    "cancel_transfer",
    "top_up_limits",
    "wrong_amount_of_cash_received",
    "card_payment_fee_charged",
    "transfer_not_received_by_recipient",
    "supported_cards_and_currencies",
    "getting_virtual_card",
    "card_acceptance",
    "top_up_reverted",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "card_payment_not_recognised",
    "edit_personal_details",
    "why_verify_identity",
    "unable_to_verify_identity",
    "get_physical_card",
    "visa_or_mastercard",
    "topping_up_by_card",
    "disposable_card_limits",
    "compromised_card",
    "atm_support",
    "direct_debit_payment_not_recognised",
    "passcode_forgotten",
    "declined_cash_withdrawal",
    "pending_card_payment",
    "lost_or_stolen_phone",
    "request_refund",
    "declined_transfer",
    "Refund_not_showing_up",
    "declined_card_payment",
    "pending_transfer",
    "terminate_account",
    "card_swallowed",
    "transaction_charged_twice",
    "verify_source_of_funds",
    "transfer_timing",
    "reverted_card_payment?",
    "change_pin",
    "beneficiary_not_allowed",
    "transfer_fee_charged",
    "receiving_money",
    "failed_transfer",
    "transfer_into_account",
    "verify_top_up",
    "getting_spare_card",
    "top_up_by_cash_or_cheque",
    "order_physical_card",
    "virtual_card_not_working",
    "wrong_exchange_rate_for_cash_withdrawal",
    "get_disposable_virtual_card",
    "top_up_failed",
    "balance_not_updated_after_bank_transfer",
    "cash_withdrawal_not_recognised",
    "exchange_charge",
    "top_up_by_card_charge",
    "activate_my_card",
    "cash_withdrawal_charge",
    "card_about_to_expire",
    "apple_pay_or_google_pay",
    "verify_my_identity",
    "country_support",
]


def config():
    return load_experiment_config(CONFIG_PATH)


def with_options(base, options):
    assert base.question is not None
    return replace(base, question=replace(base.question, options=tuple(options)))


def test_builds_frozen_choice_and_state_from_official_inventory():
    task = build_banking77_task(config(), LABELS)

    assert task.state_field == "customer_message"
    assert task.question_id == "intent"
    assert task.build_state("Where is my card?") == {"customer_message": "Where is my card?"}
    assert isinstance(task.choice, Choice)
    assert task.choice.instructions == INSTRUCTIONS
    assert list(task.choice.criteria) == LABELS
    assert task.questions() == {"intent": task.choice}

    for criterion in task.choice.criteria.values():
        assert set(criterion) == {"use_when", "distinguish_from"}
        assert all(isinstance(value, str) and value.strip() for value in criterion.values())

    assert "Refund_not_showing_up" in task.choice.criteria
    assert "reverted_card_payment?" in task.choice.criteria
    assert "PIN" in task.choice.criteria["get_physical_card"]["use_when"]
    assert "order_physical_card" in task.choice.criteria["get_physical_card"]["distinguish_from"]


def test_rejects_missing_duplicate_unexpected_and_reordered_labels():
    base = config()
    assert base.question is not None
    missing = with_options(base, base.question.options[:-1])
    with pytest.raises(ValueError, match="missing=.*country_support"):
        build_banking77_task(missing, LABELS)

    duplicate_options = list(base.question.options)
    duplicate_options[-1] = replace(duplicate_options[-1], label=LABELS[0])
    with pytest.raises(ValueError, match="Duplicate.*card_arrival"):
        build_banking77_task(with_options(base, duplicate_options), LABELS)

    unexpected_options = list(base.question.options)
    unexpected_options[-1] = replace(unexpected_options[-1], label="other")
    with pytest.raises(ValueError, match="unexpected=.*other"):
        build_banking77_task(with_options(base, unexpected_options), LABELS)

    reordered_options = list(base.question.options)
    reordered_options[0:2] = reversed(reordered_options[0:2])
    with pytest.raises(ValueError, match="label order"):
        build_banking77_task(with_options(base, reordered_options), LABELS)


def test_rejects_invalid_state_question_and_expected_inventory():
    base = config()
    assert base.question is not None
    task = build_banking77_task(base, LABELS)
    with pytest.raises(ValueError, match="nonempty"):
        task.build_state("")

    invalid_state = replace(base, question=replace(base.question, state_field="message"))
    with pytest.raises(ValueError, match="customer_message"):
        build_banking77_task(invalid_state, LABELS)

    invalid_question = replace(base, question=replace(base.question, question_id="category"))
    with pytest.raises(ValueError, match="intent"):
        build_banking77_task(invalid_question, LABELS)

    with pytest.raises(ValueError, match="77 unique"):
        build_banking77_task(base, [*LABELS[:-1], LABELS[0]])
