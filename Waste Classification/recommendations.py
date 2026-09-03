import argparse


RECOMMENDATIONS = {
    "Hazardous": {
        "title": "Hazardous Waste",
        "advice": (
            "Do not put this item in normal household waste. "
            "Keep it separate and take it to an authorized hazardous-waste "
            "or e-waste collection centre when applicable."
        ),
    },
    "Non-Recyclable": {
        "title": "Non-Recyclable Waste",
        "advice": (
            "Place this item in the general or non-recyclable waste bin. "
            "Do not place contaminated waste in the recycling bin."
        ),
    },
    "Organic": {
        "title": "Organic Waste",
        "advice": (
            "Place this item in an organic-waste bin, compost it at home, "
            "or send it to a composting facility."
        ),
    },
    "Recyclable": {
        "title": "Recyclable Waste",
        "advice": (
            "Clean and dry the item if possible, then place it in the "
            "recycling bin or take it to a local recycling collection point."
        ),
    },
}


def get_recommendation(category):
    """Return disposal guidance for a predicted waste category."""
    return RECOMMENDATIONS.get(
        category,
        {
            "title": "Unknown Waste Type",
            "advice": (
                "Unable to provide disposal guidance. Please check with "
                "your local waste-management authority."
            ),
        },
    )


def display_recommendation(category):
    """Print disposal guidance neatly in the terminal."""
    recommendation = get_recommendation(category)

    print("\n" + "=" * 55)
    print("DISPOSAL RECOMMENDATION")
    print("=" * 55)

    print(f"\nCategory: {recommendation['title']}")
    print(f"Advice: {recommendation['advice']}")

    print("\n" + "=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Display disposal guidance for a waste category."
    )

    parser.add_argument(
        "--category",
        required=True,
        choices=list(RECOMMENDATIONS.keys()),
        help="Waste category.",
    )

    arguments = parser.parse_args()

    display_recommendation(arguments.category)


if __name__ == "__main__":
    main()