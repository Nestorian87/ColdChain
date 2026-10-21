// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;
// ABI design only. No implementation or on-chain test completion is claimed.
enum ShipmentState { Funded, InTransit, Delivered, Proposed, Disputed, Settled, Cancelled }
struct Terms {
    uint256 agreedFee;
    int16 minTempDeciC; int16 maxTempDeciC;
    uint32 breachLimitSeconds; uint16 penaltyBps; uint16 riskThresholdBps;
    uint16 minValidBps; uint32 sampleStepSeconds; uint32 maxGapSeconds;
    uint64 startBy; uint64 deadline; bytes32 modelHash;
}
struct Batch { address supplier; address custodian; bytes32 metadataHash; }
struct Evidence {
    bytes32 proofHash; bytes32 modelHash; bytes32 featuresHash;
    uint16 scoreBps; uint16 validBps; uint32 breachSeconds; uint32 maxGapSeconds;
    uint64 observedUntil; uint64 from; uint64 to; bool closed;
}
struct Shipment {
    uint256 batchId; address buyer; address carrier; uint256 fee;
    ShipmentState state; Terms terms; uint64 startedAt;
    uint64 deliveredAt; uint64 challengeUntil; uint64 disputedAt;
    uint256 buyerCredit; uint256 carrierCredit;
}
interface IBatchRegistry {
    event BatchRegistered(uint256 indexed batchId,address indexed supplier,bytes32 metadataHash);
    event CustodyProposed(uint256 indexed batchId,address indexed recipient);
    event CustodyAccepted(uint256 indexed batchId,address indexed custodian);
    function registerBatch(bytes32 metadataHash) external returns(uint256);
    function proposeCustody(uint256 batchId,address recipient) external;
    function acceptCustody(uint256 batchId) external;
    function getBatch(uint256 batchId) external view returns(Batch memory);
}
interface ITelemetryRegistry {
    event PacketCommitted(uint256 indexed shipmentId,uint64 seq,bytes32 packetHash);
    event ForecastSubmitted(uint256 indexed shipmentId,uint16 scoreBps,bytes32 modelHash);
    event TelemetryClosed(uint256 indexed shipmentId,bytes32 proofHash);
    function commitPacket(uint256 id,uint64 seq,bytes32 hash,uint64 from,uint64 to,uint16 validBps) external;
    function submitForecast(uint256 id,uint16 scoreBps,bytes32 modelHash,bytes32 featuresHash,uint64 observedUntil) external;
    function closeTelemetry(uint256 id,bytes32 proofHash,uint32 breachSeconds,uint16 validBps,uint32 maxGapSeconds,uint64 from,uint64 to) external;
    function getEvidence(uint256 id) external view returns(Evidence memory);
}
interface IShipmentEscrow {
    event ShipmentCreated(uint256 indexed shipmentId,uint256 indexed batchId,address buyer,address carrier);
    event ShipmentStarted(uint256 indexed shipmentId,uint64 startedAt);
    event Delivered(uint256 indexed shipmentId,uint64 deliveredAt);
    event SettlementProposed(uint256 indexed shipmentId,uint256 buyerCredit,uint256 carrierCredit,uint64 challengeUntil);
    event Disputed(uint256 indexed shipmentId,bytes32 reasonHash);
    event Settled(uint256 indexed shipmentId,uint256 buyerCredit,uint256 carrierCredit);
    event Withdrawn(address indexed account,uint256 amount);
    function configureTelemetry(address telemetry) external;
    function createShipment(uint256 batchId,address carrier,Terms calldata terms) external payable returns(uint256);
    function startShipment(uint256 id) external;
    function markDelivered(uint256 id) external;
    function proposeSettlement(uint256 id) external;
    function challenge(uint256 id,bytes32 reasonHash) external;
    function finalize(uint256 id) external;
    function resolveDispute(uint256 id,uint16 carrierBps) external;
    function timeout(uint256 id) external;
    function withdraw() external;
    function getShipment(uint256 id) external view returns(Shipment memory);
}
