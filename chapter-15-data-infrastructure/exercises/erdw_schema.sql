-- ERDW (Enterprise Risk Data Warehouse) schema DDL — Exercise 15.2
-- BCBS 239 P6 test scenario: credit exposure by industry sector.
-- Synthetic dataset: ~28,000 CREDIT_EXPOSURE rows across three divisions.

CREATE TABLE ENTITY_MASTER (
    awb_customer_id   VARCHAR(20)  PRIMARY KEY,
    legal_name        VARCHAR(200) NOT NULL,
    sic_code          CHAR(5)      NOT NULL,   -- 2-digit sector = LEFT(sic_code, 2)
    division          VARCHAR(20)  NOT NULL
        CHECK (division IN ('CORPORATE', 'COMMERCIAL', 'RETAIL_BUSINESS')),
    country_code      CHAR(2)      NOT NULL DEFAULT 'GB',
    lei               CHAR(20)
);

CREATE TABLE FACILITY_TERMS (
    facility_id       VARCHAR(20)  PRIMARY KEY,
    awb_customer_id   VARCHAR(20)  NOT NULL REFERENCES ENTITY_MASTER,
    facility_type     VARCHAR(30)  NOT NULL,   -- TERM_LOAN | RCF | OVERDRAFT ...
    committed_gbp     NUMERIC(18,2) NOT NULL,
    netting_agreement BOOLEAN      NOT NULL DEFAULT FALSE,  -- CRR3 on-BS netting
    start_date        DATE         NOT NULL,
    maturity_date     DATE         NOT NULL
);

CREATE TABLE CREDIT_EXPOSURE (
    exposure_id       BIGINT       PRIMARY KEY,
    facility_id       VARCHAR(20)  NOT NULL REFERENCES FACILITY_TERMS,
    reporting_date    DATE         NOT NULL,
    drawn_gbp         NUMERIC(18,2) NOT NULL,
    undrawn_gbp       NUMERIC(18,2) NOT NULL,
    cash_collateral_gbp NUMERIC(18,2) NOT NULL DEFAULT 0,   -- nettable when
    provision_gbp     NUMERIC(18,2) NOT NULL DEFAULT 0      -- netting_agreement
);

CREATE INDEX ix_exposure_date ON CREDIT_EXPOSURE (reporting_date);
CREATE INDEX ix_entity_sector ON ENTITY_MASTER (sic_code);

-- P6 reference answer shape (see solutions/ex2_bcbs239_p6.py):
--   sector (2-digit SIC) | division | total_exposure_gbp
-- Netting rule: exposure = drawn - CASE WHEN netting_agreement
--   THEN cash_collateral ELSE 0 END  (floored at zero per CRR3 Art. 219).
