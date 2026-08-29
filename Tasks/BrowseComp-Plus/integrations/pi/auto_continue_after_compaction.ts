const CONTINUATION_PROMPT = `[BrowseComp automatic continuation]
Context compaction finished while the original research task was still incomplete.
Continue from the retained summary. Resume the next concrete search or document
inspection; do not treat compaction or this message as task completion. Stop only
after producing the required Explanation, Exact Answer, and Confidence response.`;

function textContent(content) {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((part) => part && part.type === "text" && typeof part.text === "string")
		.map((part) => part.text)
		.join("\n");
}

function hasCompleteBrowseCompAnswer(text) {
	return /(?:^|\n)\s*Exact Answer:\s*\S/i.test(text) && /(?:^|\n)\s*Confidence:\s*\S/i.test(text);
}

export default function (pi) {
	let lastAssistantText = "";

	pi.on("message_end", (event) => {
		if (event.message?.role !== "assistant") return;
		lastAssistantText = textContent(event.message.content);
	});

	pi.on("session_compact", (event) => {
		const enabled = !/^(?:0|false|no)$/i.test(
			process.env.BROWSECOMP_PI_AUTO_CONTINUE_COMPACTION || "1",
		);
		if (!enabled || event.reason !== "threshold" || event.willRetry) return;
		if (hasCompleteBrowseCompAnswer(lastAssistantText)) return;

		// session_compact fires while AgentSession is still streaming. Queueing a
		// follow-up here makes Pi's post-compaction hasQueuedMessages() check true,
		// so --print continues instead of emitting agent_settled immediately.
		pi.sendUserMessage(CONTINUATION_PROMPT, { deliverAs: "followUp" });
	});
}
