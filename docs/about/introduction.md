# About

**knowledge** is an AI developer assistant for the Kalisio ecosystem. It indexes the corpus into a vector database ([Qdrant](https://qdrant.tech/)) and answers questions over it.

![architecture overwiew](/images/architecture-overwiew.png)

All of it is served over HTTP, so a client retrieves only the chunks relevant to a question.
