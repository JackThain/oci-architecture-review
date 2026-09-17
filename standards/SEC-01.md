# SEC-01 Encryption with customer-managed keys
All data stores holding client data (databases, object storage buckets, block volumes)
must be encrypted with customer-managed keys held in the cloud provider's key
management service (OCI Vault, AWS KMS).
Provider-managed default keys are not sufficient for confidential client data.
Keywords: encryption, encrypted, keys, vault, bucket, storage, database, volume
