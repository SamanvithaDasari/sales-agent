"""Quick check: do the CRM tools work without involving any LLM?"""

from tools.crm_tools import (
    find_account_by_name,
    get_deals_for_account,
    get_contacts_for_account,
    get_recent_activities_for_account,
)

print("=" * 60)
print("TEST 1: Find account by partial name")
print("=" * 60)
# .func calls the underlying function directly, bypassing the agent wrapper
result = find_account_by_name.func("Davis")
print(result[:500])
print()

# Pick the first matching account id from the JSON output
import json
accounts = json.loads(result)
if not accounts:
    print("❌ No accounts found matching 'Davis'. Did you run seed_crm.py?")
    exit(1)

account_id = accounts[0]["id"]
account_name = accounts[0]["name"]
print(f"Using account_id={account_id} ({account_name}) for the next tests\n")

print("=" * 60)
print(f"TEST 2: Deals for {account_name}")
print("=" * 60)
print(get_deals_for_account.func(account_id)[:600])
print()

print("=" * 60)
print(f"TEST 3: Contacts at {account_name}")
print("=" * 60)
print(get_contacts_for_account.func(account_id)[:600])
print()

print("=" * 60)
print(f"TEST 4: Recent activities for {account_name}")
print("=" * 60)
print(get_recent_activities_for_account.func(account_id, limit=5)[:800])