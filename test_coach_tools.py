"""Smoke test the Deal Coach tools without involving Gemini."""

from tools.coach_tools import (
    get_deal_details,
    get_deal_activities,
    find_similar_deal_activities,
    get_deals_at_stage,
    get_won_lost_outcomes,
)

print("=" * 60)
print("TEST 1: Get deal details for 'Streamline Magnetic'")
print("=" * 60)
result = get_deal_details("Streamline Magnetic")
print(result[:500])

import json
deals = json.loads(result)
deal_id = deals[0]["id"] if deals else None
print(f"\nFirst deal id: {deal_id}\n")

if deal_id:
    print("=" * 60)
    print(f"TEST 2: Activities for deal {deal_id}")
    print("=" * 60)
    print(get_deal_activities(deal_id)[:500])
    print()

print("=" * 60)
print("TEST 3: Similar to 'pricing pushback on a stuck deal'")
print("=" * 60)
print(find_similar_deal_activities("pricing pushback on a stuck deal", k=3)[:600])
print()

print("=" * 60)
print("TEST 4: Deals at stage 'negotiation'")
print("=" * 60)
print(get_deals_at_stage("negotiation", limit=3)[:500])
print()

print("=" * 60)
print("TEST 5: Won vs Lost for 'discount negotiation'")
print("=" * 60)
print(get_won_lost_outcomes("discount negotiation", k=2))