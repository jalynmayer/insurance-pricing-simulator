
DROP TABLE IF EXISTS policyholders;
CREATE TABLE policyholders (
    policy_id INTEGER PRIMARY KEY,
    age INTEGER,
    vehicle_type TEXT,
    location TEXT,
    exposure REAL
);

DROP TABLE IF EXISTS claims;
CREATE TABLE claims (
    claim_id INTEGER PRIMARY KEY,
    policy_id INTEGER,
    claim_count INTEGER,
    claim_amount REAL,
    FOREIGN KEY(policy_id) REFERENCES policyholders(policy_id)
);


