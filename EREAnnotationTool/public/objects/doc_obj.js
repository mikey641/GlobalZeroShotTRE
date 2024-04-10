class DocObject {
    constructor() {
        this.doc_id = null;
        this.tokens = null;
        this.mentions = null;
    }

    static fromJsonObject(jsonObject) {
        let retDoc = new DocObject();
        if (jsonObject != null) {
            if('doc_id' in jsonObject) retDoc.doc_id = jsonObject.doc_id;
            if(jsonObject.tokens) retDoc.tokens = jsonObject.tokens;

            retDoc.mentions = [];
            if(jsonObject.mentions) {
                for (let i = 0; i < jsonObject.mentions.length; i++) {
                    retDoc.mentions.push(EventObject.fromJsonObject(jsonObject.mentions[i]));
                }
            }
        } else {
            throw new Error('Trying to create event from null object');
        }

        return retDoc;
    }
}