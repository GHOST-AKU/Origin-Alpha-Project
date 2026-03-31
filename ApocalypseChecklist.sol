// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ApocalypseChecklist
/// @notice 一个娱乐性质的“世界毁灭待办”合约，链上仅做记录与触发事件，不会真的执行现实操作。
contract ApocalypseChecklist {
    address public owner;
    bool public worldEnded;

    event WorldEndDeclared(uint256 timestamp, string message);
    event ActionTriggered(uint256 indexed actionId, string actionName, uint256 timestamp);

    error NotOwner();
    error WorldNotEnded();
    error AlreadyDeclared();

    struct Action {
        uint256 id;
        string name;
        bool done;
        uint256 doneAt;
    }

    Action[] private actions;

    constructor() {
        owner = msg.sender;

        actions.push(Action({id: 1, name: "下载原神", done: false, doneAt: 0}));
        actions.push(Action({id: 2, name: "删除COD", done: false, doneAt: 0}));
        actions.push(Action({id: 3, name: "在时代广场大屏幕播放NDX100BEST", done: false, doneAt: 0}));
    }

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    /// @notice 宣布“世界毁灭”状态（仅可调用一次）
    function declareWorldEnd() external onlyOwner {
        if (worldEnded) revert AlreadyDeclared();
        worldEnded = true;
        emit WorldEndDeclared(block.timestamp, "世界毁灭模式已开启");
    }

    /// @notice 标记某个待办事项已触发（需要先宣布 worldEnded）
    function triggerAction(uint256 actionId) external onlyOwner {
        if (!worldEnded) revert WorldNotEnded();
        require(actionId > 0 && actionId <= actions.length, "Invalid actionId");

        Action storage action = actions[actionId - 1];
        action.done = true;
        action.doneAt = block.timestamp;

        emit ActionTriggered(action.id, action.name, block.timestamp);
    }

    function getAction(uint256 actionId)
        external
        view
        returns (uint256 id, string memory name, bool done, uint256 doneAt)
    {
        require(actionId > 0 && actionId <= actions.length, "Invalid actionId");
        Action memory action = actions[actionId - 1];
        return (action.id, action.name, action.done, action.doneAt);
    }

    function getAllActions() external view returns (Action[] memory) {
        return actions;
    }
}
