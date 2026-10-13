"""
word_predictor.py
-----------------
Frequency-weighted word prediction and autocomplete engine for the Assistive Eye-Gaze Keyboard.
Provides instant prefix matching (<1ms) using top common English words and assistive emergency phrases.
"""

# Top frequently used English words for fast prediction
COMMON_WORDS = [
    # Top 1-100
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
    "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up",
    "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
    "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "new", "want", "because", "any", "these", "give", "day", "most", "us",

    # Top 101-250
    "is", "are", "was", "were", "been", "has", "had", "water", "food", "help", "pain", "doctor",
    "please", "thank", "thanks", "yes", "no", "okay", "need", "feel", "hungry", "thirsty", "tired",
    "cold", "hot", "medicine", "nurse", "family", "call", "stop", "start", "more", "less", "right",
    "left", "up", "down", "open", "close", "turn", "light", "bed", "chair", "room", "phone", "tv",
    "music", "read", "book", "talk", "listen", "sleep", "rest", "sit", "stand", "walk", "great",
    "fine", "bad", "happy", "sad", "love", "friend", "home", "today", "tomorrow", "yesterday",
    "morning", "night", "always", "never", "maybe", "why", "where", "how", "again", "much", "many",
    "very", "too", "quite", "really", "almost", "enough", "such", "same", "different", "small", "big",
    "large", "long", "short", "high", "low", "far", "near", "early", "late", "next", "last", "few",

    # Additional vocabulary
    "able", "about", "above", "across", "action", "activity", "actually", "add", "address", "adult",
    "afraid", "afternoon", "again", "against", "age", "agree", "air", "allow", "almost", "alone",
    "along", "already", "alright", "also", "although", "always", "am", "among", "amount", "another",
    "answer", "anybody", "anymore", "anyone", "anything", "anyway", "anywhere", "apartment", "appear",
    "apply", "area", "arm", "around", "arrive", "art", "article", "ask", "assist", "assistance",
    "attention", "aunt", "available", "away", "baby", "back", "bad", "bag", "ball", "bank", "bar",
    "base", "basic", "bath", "bathroom", "battery", "beautiful", "become", "bed", "bedroom", "before",
    "begin", "behind", "believe", "bell", "belong", "below", "beside", "best", "better", "between",
    "beyond", "bicycle", "bill", "bird", "birth", "birthday", "bit", "bite", "black", "blanket",
    "blind", "blood", "blue", "board", "boat", "body", "bone", "bottle", "bottom", "bowl", "box",
    "boy", "brain", "bread", "break", "breakfast", "breath", "breathe", "bright", "bring", "brother",
    "brown", "brush", "build", "building", "burn", "bus", "business", "busy", "button", "buy",
    "cake", "calendar", "call", "camera", "camp", "can", "candle", "car", "card", "care", "careful",
    "carry", "case", "cat", "catch", "cause", "cell", "chair", "chance", "change", "charge", "check",
    "cheese", "chest", "chicken", "child", "children", "choose", "church", "circle", "city", "class",
    "clean", "clear", "climb", "clock", "close", "clothes", "cloud", "coat", "coffee", "cold", "color",
    "come", "comfort", "comfortable", "common", "company", "complete", "computer", "condition",
    "continue", "control", "cook", "cool", "copy", "corner", "correct", "cost", "couch", "cough",
    "count", "country", "couple", "course", "court", "cover", "crazy", "cream", "cross", "cry",
    "cup", "cure", "cut", "cute", "dad", "daily", "dance", "danger", "dark", "date", "daughter",
    "dawn", "dead", "deal", "dear", "death", "decide", "deep", "degree", "delicious", "dentist",
    "department", "describe", "desk", "device", "diet", "dinner", "direction", "dirty", "discover",
    "disease", "dish", "distance", "do", "doctor", "dog", "door", "doubt", "draw", "dream", "dress",
    "drink", "drive", "driver", "drop", "drug", "dry", "due", "during", "dust", "duty", "each",
    "ear", "early", "earn", "earth", "ease", "east", "easy", "eat", "edge", "education", "effect",
    "effort", "egg", "eight", "either", "electric", "elevator", "else", "emergency", "empty", "end",
    "energy", "engine", "enjoy", "enough", "enter", "entire", "environment", "equal", "escape",
    "especially", "evening", "event", "ever", "every", "everybody", "everyone", "everything",
    "everywhere", "exact", "exam", "example", "excellent", "except", "exercise", "exist", "expect",
    "expensive", "experience", "explain", "eye", "face", "fact", "fail", "fair", "fall", "false",
    "family", "famous", "fan", "far", "farm", "fast", "fat", "father", "fault", "favorite", "fear",
    "feed", "feel", "feeling", "feet", "female", "fever", "field", "fight", "figure", "fill", "film",
    "final", "find", "fine", "finger", "finish", "fire", "first", "fish", "fit", "five", "fix",
    "flat", "floor", "flower", "fly", "focus", "follow", "food", "foot", "for", "force", "foreign",
    "forest", "forever", "forget", "forgive", "fork", "form", "forward", "four", "free", "fresh",
    "friend", "from", "front", "fruit", "full", "fun", "future", "game", "garden", "gas", "gate",
    "gather", "general", "gentle", "get", "gift", "girl", "give", "glad", "glass", "glasses", "go",
    "goal", "gold", "good", "goodbye", "government", "grab", "grade", "grandfather", "grandmother",
    "grass", "great", "green", "greet", "grey", "ground", "group", "grow", "guard", "guess", "guest",
    "guide", "gun", "guy", "habit", "hair", "half", "hall", "hand", "handle", "hang", "happen",
    "happy", "hard", "hardly", "hat", "hate", "have", "head", "headache", "health", "healthy",
    "hear", "heart", "heat", "heavy", "hello", "help", "helpful", "here", "hero", "hide", "high",
    "hill", "history", "hit", "hold", "hole", "holiday", "home", "honest", "hope", "hospital", "hot",
    "hotel", "hour", "house", "how", "however", "huge", "human", "hungry", "hunt", "hurry", "hurt",
    "husband", "ice", "idea", "identify", "ill", "image", "imagine", "important", "improve", "in",
    "inch", "include", "increase", "indeed", "inform", "information", "injure", "inside", "instead",
    "interest", "internet", "interview", "into", "introduce", "invent", "invite", "iron", "island",
    "issue", "it", "item", "jacket", "job", "join", "joke", "journey", "joy", "judge", "juice",
    "jump", "just", "keep", "key", "kick", "kid", "kill", "kind", "king", "kiss", "kitchen", "knee",
    "knife", "knock", "know", "knowledge", "lady", "lake", "lamp", "land", "language", "large",
    "last", "late", "later", "laugh", "law", "lawyer", "lay", "lazy", "lead", "leader", "leaf",
    "learn", "least", "leave", "left", "leg", "lesson", "let", "letter", "level", "library", "lie",
    "life", "lift", "light", "like", "likely", "limit", "line", "lip", "liquid", "list", "listen",
    "little", "live", "living", "local", "lock", "lonely", "long", "look", "lose", "loss", "lost",
    "lot", "loud", "love", "low", "luck", "lucky", "lunch", "machine", "mad", "magazine", "mail",
    "main", "major", "make", "male", "man", "manage", "many", "map", "mark", "market", "marry",
    "match", "matter", "may", "maybe", "meal", "mean", "meaning", "measure", "meat", "medicine",
    "meet", "meeting", "member", "memory", "mention", "message", "metal", "middle", "might", "milk",
    "mind", "minute", "mirror", "miss", "mistake", "mix", "mobile", "model", "modern", "moment",
    "money", "month", "mood", "moon", "more", "morning", "most", "mother", "motion", "motor",
    "mountain", "mouth", "move", "movie", "much", "muscle", "music", "must", "name", "narrow",
    "nation", "natural", "nature", "near", "neat", "necessary", "neck", "need", "needle", "neighbor",
    "neither", "nerve", "nervous", "net", "never", "new", "news", "next", "nice", "night", "nine",
    "no", "nobody", "nod", "noise", "none", "noon", "nor", "normal", "north", "nose", "not", "note",
    "nothing", "notice", "now", "number", "nurse", "nut", "object", "observe", "obtain", "obvious",
    "occur", "ocean", "odd", "off", "offer", "office", "officer", "often", "oil", "okay", "old",
    "on", "once", "one", "only", "onto", "open", "operate", "operation", "opinion", "opportunity",
    "opposite", "order", "ordinary", "organize", "original", "other", "otherwise", "ought", "our",
    "out", "outside", "over", "own", "owner", "page", "pain", "paint", "pair", "palace", "pale",
    "paper", "parent", "park", "part", "partner", "party", "pass", "passage", "past", "path",
    "patient", "pattern", "pay", "peace", "pen", "pencil", "people", "per", "perfect", "perform",
    "period", "person", "personal", "phone", "photo", "physical", "pick", "picture", "piece",
    "pillow", "place", "plain", "plan", "plane", "plant", "plastic", "plate", "play", "player",
    "pleasant", "please", "pleasure", "plenty", "pocket", "point", "police", "polite", "pool",
    "poor", "popular", "position", "positive", "possible", "post", "potato", "power", "practice",
    "pray", "prefer", "prepare", "present", "president", "press", "pressure", "pretty", "prevent",
    "price", "pride", "primary", "print", "private", "prize", "probably", "problem", "produce",
    "product", "program", "progress", "promise", "proper", "protect", "proud", "prove", "provide",
    "public", "pull", "pump", "pure", "purpose", "push", "put", "quality", "quarter", "queen",
    "question", "quick", "quiet", "quite", "radio", "rain", "raise", "range", "rare", "rate",
    "rather", "raw", "reach", "react", "read", "ready", "real", "realize", "really", "reason",
    "receive", "recent", "recognize", "record", "recover", "red", "reduce", "refuse", "regard",
    "regular", "relative", "relax", "remain", "remember", "remind", "remove", "repeat", "replace",
    "reply", "report", "request", "require", "rescue", "respect", "respond", "rest", "restaurant",
    "result", "return", "rich", "ride", "right", "ring", "rise", "risk", "river", "road", "rock",
    "roll", "room", "root", "rope", "rough", "round", "route", "row", "rule", "run", "rush",
    "safe", "safety", "sail", "sale", "salt", "same", "sample", "sand", "save", "say", "scale",
    "scared", "scene", "schedule", "school", "science", "score", "screen", "sea", "search", "season",
    "seat", "second", "secret", "section", "see", "seed", "seek", "seem", "select", "self", "sell",
    "send", "sense", "sentence", "separate", "serious", "serve", "service", "set", "settle", "seven",
    "several", "shake", "shall", "shape", "share", "sharp", "she", "sheet", "shelf", "shine",
    "ship", "shirt", "shoe", "shoot", "shop", "short", "should", "shoulder", "shout", "show",
    "shower", "shut", "sick", "side", "sight", "sign", "signal", "silence", "silent", "silver",
    "simple", "since", "sing", "single", "sister", "sit", "site", "situation", "six", "size",
    "skill", "skin", "skirt", "sky", "sleep", "sleepy", "slide", "slight", "slip", "slow", "small",
    "smart", "smell", "smile", "smoke", "smooth", "snake", "snow", "so", "soap", "social", "society",
    "sock", "soft", "soil", "solid", "some", "somebody", "someone", "something", "sometimes",
    "somewhere", "son", "song", "soon", "sore", "sorry", "sound", "soup", "sour", "south", "space",
    "spare", "speak", "special", "specific", "speed", "spell", "spend", "spill", "spin", "spite",
    "split", "spoon", "sport", "spot", "spread", "spring", "square", "stage", "stair", "stamp",
    "stand", "standard", "star", "stare", "start", "state", "station", "stay", "steal", "steam",
    "step", "stick", "still", "stomach", "stone", "stop", "store", "storm", "story", "straight",
    "strange", "street", "strength", "stretch", "strike", "strong", "structure", "struggle", "student",
    "study", "stuff", "style", "subject", "substance", "succeed", "success", "such", "sudden",
    "suffer", "sugar", "suggest", "suit", "summer", "sun", "supper", "supply", "support", "suppose",
    "sure", "surface", "surprise", "surround", "sweet", "swim", "switch", "system", "table",
    "take", "talk", "tall", "taste", "tax", "tea", "teach", "teacher", "team", "tear", "telephone",
    "tell", "temperature", "ten", "tend", "term", "terrible", "test", "than", "thank", "thanks",
    "that", "the", "their", "them", "then", "there", "therefore", "these", "they", "thick", "thin",
    "thing", "think", "third", "thirst", "thirsty", "this", "though", "thought", "thousand", "three",
    "throat", "through", "throw", "thumb", "ticket", "tide", "tie", "tight", "till", "time", "tiny",
    "tip", "tire", "tired", "title", "to", "today", "toe", "together", "toilet", "tomorrow", "tonight",
    "too", "tool", "tooth", "top", "total", "touch", "tough", "toward", "towel", "tower", "town",
    "track", "trade", "traffic", "train", "travel", "treat", "tree", "trip", "trouble", "true",
    "trust", "truth", "try", "tube", "turn", "twelve", "twenty", "twice", "two", "type", "ugly",
    "uncle", "under", "understand", "unit", "until", "up", "upon", "upper", "upset", "urgent",
    "use", "useful", "usual", "vacation", "valley", "valuable", "value", "various", "vegetable",
    "vehicle", "very", "victim", "view", "village", "visit", "visitor", "voice", "volume", "wait",
    "wake", "walk", "wall", "want", "war", "warm", "warn", "wash", "waste", "watch", "water",
    "wave", "way", "we", "weak", "wear", "weather", "wedding", "week", "weigh", "weight", "welcome",
    "well", "west", "wet", "what", "whatever", "wheel", "when", "whenever", "where", "wherever",
    "whether", "which", "while", "white", "who", "whole", "whom", "whose", "why", "wide", "wife",
    "wild", "will", "win", "wind", "window", "wine", "wing", "winter", "wipe", "wire", "wise",
    "wish", "with", "within", "without", "woman", "wonder", "wonderful", "wood", "wooden", "word",
    "work", "worker", "world", "worry", "worse", "worst", "worth", "would", "wound", "wrap",
    "wrist", "write", "writer", "wrong", "yard", "yeah", "year", "yellow", "yes", "yesterday",
    "yet", "you", "young", "your", "yours", "yourself", "zero", "zone"
]

# Quick assistive / emergency phrases for one-glance communication
QUICK_ASSIST_PHRASES = [
    "I need help",
    "Water please",
    "I am in pain",
    "Thank you",
    "Yes",
    "No",
    "I am hungry",
    "Call doctor",
    "Please adjust bed",
    "I am tired"
]

class WordPredictor:
    """
    Fast prefix prediction engine.
    Given a partial word (prefix), returns the top candidate words ordered by likelihood.
    """
    def __init__(self, vocabulary=None):
        raw_vocab = vocabulary if vocabulary is not None else COMMON_WORDS
        # Deduplicate while preserving order of frequency
        seen = set()
        self.words = []
        for w in raw_vocab:
            w_clean = w.strip().lower()
            if w_clean and w_clean not in seen:
                seen.add(w_clean)
                self.words.append(w_clean)

    def predict(self, prefix, max_results=4):
        """
        Returns up to max_results words that start with prefix.
        If prefix is empty, returns common high-frequency starters.
        """
        prefix = prefix.strip().lower()
        if not prefix:
            return ["I", "the", "need", "please"][:max_results]

        matches = []
        for word in self.words:
            if word.startswith(prefix) and word != prefix:
                matches.append(word)
                if len(matches) >= max_results:
                    break

        # If not enough matches, pad with capitalized prefix if applicable
        if not matches:
            matches = [prefix]
        return matches[:max_results]

    def get_quick_phrases(self):
        """Returns the quick emergency/assistive phrases."""
        return QUICK_ASSIST_PHRASES
