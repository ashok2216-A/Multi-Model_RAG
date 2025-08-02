Based on the content of your uploaded PDF about **Pixtral 12B**, here are **sample evaluation questions and answers** (from easy to hard) that you can use to **evaluate your RAG application**. These are drawn directly from the document’s content and can help you test retrieval, comprehension, and reasoning performance.

---

###  Easy

**Q1: What is Pixtral 12B?**
**A1:** Pixtral 12B is a 12-billion-parameter multimodal language model trained to understand both natural images and documents.

**Q2: What kind of license is Pixtral 12B released under?**
**A2:** Apache 2.0 license.

**Q3: What is the context window size of Pixtral 12B?**
**A3:** 128K tokens.

---

###  Medium

**Q4: Name two benchmark datasets where Pixtral 12B outperforms other open-source models.**
**A4:** MM-MT-Bench and MathVista.

**Q5: What is the main architectural innovation in Pixtral’s vision encoder?**
**A5:** It uses ROPE-2D (relative rotary position encoding) to handle variable image resolutions and aspect ratios.

**Q6: What is the total number of parameters in Pixtral's vision encoder?**
**A6:** 400 million parameters.

**Q7: What is the reported performance of Pixtral 12B on the MM-MT-Bench scale from LMSys-Vision?**
**A7:** 6.05 (on a scale of 1 to 10).

---

###  Hard

**Q8: How does the ROPE-2D positional encoding maintain the “relative” property?**
**A8:** It ensures that the inner product between two patch vectors depends only on their relative positional difference in height and width, not absolute positions.

**Q9: How does Pixtral handle attention leakage between patches from different images?**
**A9:** Through a block-diagonal attention mask.

**Q10: Compare Pixtral 12B’s performance on ChartQA with Qwen-2-VL 7B and LLaMA-3.2 11B under exact match metric.**
**A10:** Pixtral 12B achieves 81.8, Qwen-2-VL 7B scores 41.2, and LLaMA-3.2 11B scores 14.8.

**Q11: What were the two key issues identified with evaluation protocols for multimodal LLMs?**
**A11:** (1) Under-specified prompts that harm model performance. (2) Strict exact-match metrics penalize correct but differently formatted answers.

---

Let me know if you’d like these in JSON format for automatic evaluation, or want a UI mockup for question testing!
