"""Call a decision model directly, without a benchmark plan or labeled dataset."""

import asyncio
import json

from typesafe_sdk import Choice

from thesis_model_clients import create_decision_client, decision_result_to_dict


async def main() -> None:
    questions = {
        "intent": Choice(
            instructions="Which intent best matches the message?",
            criteria={
                "card_arrival": "The customer asks when a card will arrive.",
                "change_pin": "The customer wants to change a card PIN.",
            },
        )
    }
    async with create_decision_client(
        "system_one", "typesafe-ai/jev-1.14.0", "typesafe-ai", max_concurrency=1
    ) as client:
        result = await client.evaluate({"message": "When will my card arrive?"}, questions)
    record = decision_result_to_dict(result, answer_output_mode="native_probabilities")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
