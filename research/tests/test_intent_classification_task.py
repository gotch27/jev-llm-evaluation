from dataclasses import replace
from pathlib import Path

import pytest
from typesafe_sdk import Choice

from thesis_research.config import load_task_config
from thesis_research.datasets import ChoiceReferenceSchema
from thesis_research.tasks import build_structured_task

CONFIG_PATH = (
    Path(__file__).parents[1] / "experiments" / "intent_classification" / "tasks" / "banking77.toml"
)
WITHOUT_CRITERIA_CONFIG_PATH = CONFIG_PATH.with_name("banking77-without-criteria.toml")
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


SCHEMA = ChoiceReferenceSchema(tuple(LABELS))


def config():
    return load_task_config(CONFIG_PATH)


def with_options(base, options):
    assert base.question is not None
    return replace(base, question=replace(base.question, options=tuple(options)))


def test_builds_frozen_choice_and_state_from_official_inventory():
    task = build_structured_task(config(), SCHEMA)

    assert task.state_fields == (("customer_message", "text"),)
    assert task.question_id == "intent"
    assert task.build_state({"text": "Where is my card?"}) == {
        "customer_message": "Where is my card?"
    }
    assert isinstance(task.question, Choice)
    assert task.question.instructions == INSTRUCTIONS
    assert list(task.question.criteria) == LABELS
    assert task.questions() == {"intent": task.question}

    for criterion in task.question.criteria.values():
        assert isinstance(criterion, dict)
        assert set(criterion) == {"use_when", "distinguish_from"}
        assert all(isinstance(value, str) and value.strip() for value in criterion.values())

    assert "Refund_not_showing_up" in task.question.criteria
    assert "reverted_card_payment?" in task.question.criteria
    assert "PIN" in task.question.criteria["get_physical_card"]["use_when"]
    assert "order_physical_card" in task.question.criteria["get_physical_card"]["distinguish_from"]


def test_passes_generic_criterion_shapes_to_typesafe_choice():
    base = config()
    assert base.question is not None
    options = list(base.question.options)
    options[0] = replace(options[0], criterion="A string criterion")
    options[1] = replace(options[1], criterion=["first boundary", {"priority": 2}])
    options[2] = replace(options[2], criterion=None)

    task = build_structured_task(with_options(base, options), SCHEMA)

    assert task.question.criteria[LABELS[0]] == "A string criterion"
    assert task.question.criteria[LABELS[1]] == ["first boundary", {"priority": 2}]
    assert task.question.criteria[LABELS[2]] is None


def test_builds_without_criteria_variant_with_null_criteria():
    task = build_structured_task(load_task_config(WITHOUT_CRITERIA_CONFIG_PATH), SCHEMA)

    assert task.question.instructions == INSTRUCTIONS
    assert list(task.question.criteria) == LABELS
    assert all(criterion is None for criterion in task.question.criteria.values())


def test_rejects_missing_duplicate_unexpected_and_reordered_labels():
    base = config()
    assert base.question is not None
    missing = with_options(base, base.question.options[:-1])
    with pytest.raises(ValueError, match="missing=.*country_support"):
        build_structured_task(missing, SCHEMA)

    duplicate_options = list(base.question.options)
    duplicate_options[-1] = replace(duplicate_options[-1], label=LABELS[0])
    with pytest.raises(ValueError, match="Duplicate.*card_arrival"):
        build_structured_task(with_options(base, duplicate_options), SCHEMA)

    unexpected_options = list(base.question.options)
    unexpected_options[-1] = replace(unexpected_options[-1], label="other")
    with pytest.raises(ValueError, match="unexpected=.*other"):
        build_structured_task(with_options(base, unexpected_options), SCHEMA)

    reordered_options = list(base.question.options)
    reordered_options[0:2] = reversed(reordered_options[0:2])
    with pytest.raises(ValueError, match="label order"):
        build_structured_task(with_options(base, reordered_options), SCHEMA)


def test_state_names_and_question_ids_are_configurable():
    base = config()
    renamed = replace(
        base,
        state=replace(base.state, fields=(("message", "text"),)),
        question=replace(base.question, question_id="category"),
    )
    task = build_structured_task(renamed, SCHEMA)
    assert task.build_state({"text": "Where is my card?"}) == {"message": "Where is my card?"}
    assert task.questions() == {"category": task.question}
    with pytest.raises(ValueError, match="nonempty"):
        task.build_state({"text": ""})
    with pytest.raises(ValueError, match="missing source field: text"):
        task.build_state({"message": "Where is my card?"})
    with pytest.raises(ValueError, match="must be a mapping"):
        task.build_state("Where is my card?")
